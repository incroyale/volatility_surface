import matplotlib
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import Normalize
matplotlib.use("Qt5Agg")
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

plt.style.use('dark_background')


class sviSurface:

    def __init__(self, ticker="^SPX"):
        self.ticker = ticker
        self.calls_df = None
        self.puts_df = None
        self.iv_df = None
        self.spot_price = None

    def fetch_spot_price(self):
        return yf.Ticker(self.ticker).history(period="1d")["Close"].iloc[-1]


    def fetch_option_chain(self):
        """
        Fetches all expiries for a given ticker.
        Returns two DataFrames: calls and puts, with only selected columns:
        'strike', 'mid', 'impliedVolatility', 'inTheMoney',
        """
        all_calls = []
        all_puts = []
        ticker_obj = yf.Ticker(self.ticker)

        for expiry in ticker_obj.options:
            chain = ticker_obj.option_chain(expiry)
            calls = chain.calls[['strike', 'impliedVolatility']]
            puts = chain.puts[['strike', 'impliedVolatility']]
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            all_calls.append(calls)
            all_puts.append(puts)

        self.calls_df = pd.concat(all_calls, ignore_index=True)
        self.puts_df = pd.concat(all_puts, ignore_index=True)
        return self.calls_df, self.puts_df


    def fetch_iv_df(self, min_days=2/365, max_days=2/12, strike_width=0.1):
        """
        Combines OTM calls and puts into a single DataFrame.
        Filters expiries to [2 days, 1 year] by default.
        """
        self.calls_df['expiry'] = pd.to_datetime(self.calls_df['expiry']).dt.tz_localize('UTC')
        self.puts_df['expiry'] = pd.to_datetime(self.puts_df['expiry']).dt.tz_localize('UTC')
        today = datetime.now(timezone.utc)
        self.calls_df['T'] = (self.calls_df['expiry'] - today).dt.days / 365
        self.puts_df['T'] = (self.puts_df['expiry'] - today).dt.days / 365

        # Filter expiry range
        calls_df = self.calls_df[(self.calls_df['T'] >= min_days) & (self.calls_df['T'] <= max_days)]
        puts_df = self.puts_df[(self.puts_df['T'] >= min_days) & (self.puts_df['T'] <= max_days)]
        spot = self.fetch_spot_price()

        # Keep only OTM options
        otm_calls = calls_df[calls_df['strike'] >= spot * 0.99]
        otm_puts = puts_df[puts_df['strike'] <= spot * 1.01]

        # Filter Strike
        otm_calls = otm_calls[(otm_calls['strike'] >= (1 - strike_width) * spot) & (otm_calls['strike'] <= (1 + strike_width) * spot)]
        otm_puts = otm_puts[(otm_puts['strike'] >= (1 - strike_width) * spot) & (otm_puts['strike'] <= (1 + strike_width) * spot)]

        # Log Moneyness = ln(K/F)
        otm_calls['log_moneyness'] = np.log(otm_calls['strike'] / (spot * np.exp(0.0037 * otm_calls['T'])))
        otm_puts['log_moneyness'] = np.log(otm_puts['strike'] / (spot * np.exp(0.0037 * otm_puts['T'])))

        # IV Cleanup
        otm_calls = otm_calls[(otm_calls['impliedVolatility'] > 0.10) & (otm_calls['impliedVolatility'] < 0.5)]
        otm_puts = otm_puts[(otm_puts['impliedVolatility'] > 0.10) & (otm_puts['impliedVolatility'] < 0.5)]

        # Total Variance
        otm_calls['total_variance'] = (otm_calls['impliedVolatility'] ** 2) * (otm_calls['T'])
        otm_puts['total_variance'] = (otm_puts['impliedVolatility'] ** 2) * (otm_puts['T'])

        combined_df = pd.concat([otm_calls[['log_moneyness', 'total_variance', 'T']], otm_puts[['log_moneyness', 'total_variance', 'T']]], ignore_index=True)
        self.iv_df = combined_df
        # self.iv_df.to_csv("sample_data.csv")
        return combined_df


    def calibrate_svi_smile(self, expiry):
        """
        Fit an SVI model for one smile.
        Equation: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        """
        temp = self.iv_df[np.isclose(self.iv_df['T'], expiry)]
        temp = temp.sort_values('log_moneyness')
        # print(f"Number of points: {len(temp)}")
        # print(temp.head(20))
        # print(temp.tail(20))
        k = temp['log_moneyness'].values
        w_mkt = temp['total_variance'].values

        def svi_raw(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        def residuals(params, k, w):
            a, b, rho, m, sigma = params
            w_model = svi_raw(k, a, b, rho, m, sigma) # compute model-implied variance for all strikes
            return (w_model - w)

        a0 = np.mean(w_mkt) # initial guess
        b0 = 0.05
        rho0 = -0.5
        m0 = np.median(k)
        sigma0 = 0.2
        x0 = [a0, b0, rho0, m0, sigma0]
        # Constraints: b > 0, |rho| < 1, sigma > 0
        bounds = ([0.000, 1e-6, -0.999, -1.0, 1e-6], # Lower Bounds
                  [0.5, 1.0, 0.999, 1.0, 2.0]) # Upper Bounds
        res = least_squares(residuals, x0, args=(k, w_mkt), bounds=bounds, method='trf')
        a, b, rho, m, sigma = res.x
        k_grid = np.linspace(k.min(), k.max(), 200)
        w_fit = svi_raw(k_grid, a, b, rho, m, sigma)
        return {'k': k, 'w_mkt':w_mkt, 'w_fit':w_fit, 'k_grid':k_grid, 'params': (a, b, rho, m, sigma), 'T':expiry}


    def plot_multiple_smiles(self, num_maturities=10):
        """
        Plot multiple SVI smiles for different maturities on one figure.
        """
        expiries = sorted(self.iv_df['T'].unique())

        if len(expiries) > num_maturities: # Select evenly spaced maturities if there are too many
            indices = np.linspace(0, len(expiries) - 1, num_maturities, dtype=int)
            expiries = [expiries[i] for i in indices]

        fitted_smiles = {} # Collect all fitted smiles for surface construction

        # Plotting Code
        n_plots = len(expiries)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, expiry in enumerate(expiries):
            print(f"\nFitting T = {expiry:.3f}...")
            result = self.calibrate_svi_smile(expiry)

            if result is None:
                print(f"Skipping T = {expiry:.3f} (insufficient data)")
                axes[i].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'T = {expiry:.3f}')
                continue

            fitted_smiles[expiry] = result

            days = int(expiry * 365)
            axes[i].scatter(result['k'], result['w_mkt'], label="Market", s=25, alpha=0.6, color='cyan')
            axes[i].plot(result['k_grid'], result['w_fit'], label="SVI Fit", linewidth=2, color='lime')
            axes[i].set_xlabel('Log Moneyness')
            axes[i].set_ylabel('Total Variance')
            axes[i].set_title(f'T = {expiry:.3f} ({days} days)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            a, b, rho, m, sigma = result['params']
            print(f"  a={a:.6f}, b={b:.6f}, rho={rho:.4f}, m={m:.4f}, sigma={sigma:.4f}")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        return fitted_smiles


    def build_surface_grid(self, fitted_smiles, num_k=50, num_T=50):
        """
        Builds a 2D IV surface grid from fitted SVI smiles.

        :param fitted_smiles: dict {expiry: result} from calibrate_svi_smile()
        :param num_k: number of log_moneyness poitns
        :param num_T: number of maturities in the grid

        :return:
        k_grid: array of log_moneyness
        T_grid: array of maturities
        IV_grid: 2D array of implied vols
        """
        expiries = sorted(fitted_smiles.keys())
        a_list, b_list, rho_list, m_list, sigma_list = [], [], [], [], []
        k_min, k_max = np.inf, -np.inf

        for T in expiries:
            res = fitted_smiles[T]
            a, b, rho, m, sigma = res['params']
            a_list.append(a)
            b_list.append(b)
            rho_list.append(rho)
            m_list.append(m)
            sigma_list.append(sigma)
            k_min = min(k_min, res['k'].min())
            k_max = max(k_max, res['k'].max())

        # Interpolate each parameter through PCHIP
        a_interp = PchipInterpolator(expiries, a_list)
        b_interp = PchipInterpolator(expiries, b_list)
        rho_interp = PchipInterpolator(expiries, rho_list)
        m_interp = PchipInterpolator(expiries, m_list)
        sigma_interp = PchipInterpolator(expiries, sigma_list)

        k_grid = np.linspace(k_min, k_max, num_k)
        T_grid = np.linspace(min(expiries), max(expiries), num_T)
        IV_grid = np.zeros((num_T, num_k))

        def svi_total_variance(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        # Compute IV surface for each maturity
        for j, T in enumerate(T_grid):
            a, b, rho, m, sigma = a_interp(T), b_interp(T), rho_interp(T), m_interp(T), sigma_interp(T)
            w = svi_total_variance(k_grid, a, b, rho, m, sigma)
            w = np.clip(w, 1e-8, None)
            IV_grid[j, :] = np.sqrt(w / T) # total variance -> IV

        return k_grid, T_grid, IV_grid


if __name__ == "__main__":
    svi = sviSurface(ticker='^SPX')
    svi.fetch_option_chain()
    svi.fetch_iv_df(min_days=2/365, max_days=2/12)

    # 1) Fit IV smiles for all expiries
    fitted_smiles = svi.plot_multiple_smiles(num_maturities=8)

    # 2) Build IV surface grid
    k_grid, T_grid, IV_grid = svi.build_surface_grid(fitted_smiles, num_k=50)

    # 3) Plot 3D IV Surface (Bloomberg-style with side walls)
    Kmesh, Tmesh = np.meshgrid(k_grid, T_grid, indexing='xy')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Dark Bloomberg-style theme ----
    fig.patch.set_facecolor("#0e0f12")
    ax.set_facecolor("#0e0f12")

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.06, 0.07, 0.08, 0.0))
        pane.set_edgecolor((1, 1, 1, 0.25))

    ax.tick_params(colors="#d8d8d8")
    cmap = plt.get_cmap("turbo")
    surf = ax.plot_surface(Kmesh, Tmesh, IV_grid, cmap=cmap, linewidth=0.1, antialiased=True, alpha=0.95, rstride=1, cstride=1)
    ax.view_init(elev=30, azim=310)
    # ----- Grid styling -----
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        g = axis._axinfo["grid"]
        g["color"] = (1, 1, 1, 0.15)
        g["linewidth"] = 0.4

    # ----- Add left wall (term structure lines) -----
    k_lines = np.quantile(k_grid, np.linspace(0.05, 0.95, 6))
    norm_k = Normalize(vmin=k_grid.min(), vmax=k_grid.max())
    x_wall = k_grid.min() - 0.4 * (k_grid.max() - k_grid.min())

    for k0 in k_lines:
        idx = np.argmin(np.abs(k_grid - k0))
        iv_line = IV_grid[:, idx]
        ax.plot(T_grid, iv_line, zs=x_wall, zdir="x", color=cmap(norm_k(k0)), lw=2.0, alpha=0.95)

    # ----- Add back wall (skew lines) -----
    T_lines = np.quantile(T_grid, np.linspace(0.05, 0.95, 6))
    norm_T = Normalize(vmin=T_grid.min(), vmax=T_grid.max())
    y_back = T_grid.max() + 0.2 * (T_grid.max() - T_grid.min())

    for T0 in T_lines:
        idx = np.argmin(np.abs(T_grid - T0))
        iv_line = IV_grid[idx, :]
        ax.plot(k_grid, iv_line, zs=y_back, zdir="y", color=cmap(norm_T(T0)), lw=2.0, alpha=0.95)

    ax.set_xlabel("Log Moneyness", color="#eaeaea")
    ax.set_ylabel("Maturity (Years)", color="#eaeaea")
    ax.set_zlabel("Implied Vol", color="#eaeaea")
    ax.set_title(f"{svi.ticker} Implied Vol Surface", fontsize=16, y=1.03, color="white")
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(IV_grid)
    cbar = plt.colorbar(m, ax=ax, pad=0.1, shrink=0.5, aspect=10)
    cbar.ax.tick_params(colors="#eaeaea")
    cbar.set_label("Implied Vol", color="#eaeaea")
    plt.show()