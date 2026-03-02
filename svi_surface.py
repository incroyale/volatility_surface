import matplotlib
from scipy.interpolate import PchipInterpolator
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

plt.style.use('dark_background')
matplotlib.use("Qt5Agg")


class SviSurface:

    def __init__(self, ticker="^SPX"):
        self.ticker = ticker
        self.calls_df = None
        self.puts_df = None
        self.iv_df = None
        self.spot_price = None

    def fetch_spot_price(self):
        return yf.Ticker(self.ticker).history(period="1d")["Close"].iloc[-1]

    def fetch_risk_free_rate(self):
        try:
            tnx = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1]
            return tnx / 100
        except:
            return 0.045

    def fetch_option_chain(self):
        all_calls = []
        all_puts = []
        ticker_obj = yf.Ticker(self.ticker)
        for expiry in ticker_obj.options:
            chain = ticker_obj.option_chain(expiry)
            calls = chain.calls[['strike', 'impliedVolatility']].copy()
            puts  = chain.puts[['strike', 'impliedVolatility']].copy()
            calls['expiry'] = expiry
            puts['expiry']  = expiry
            all_calls.append(calls)
            all_puts.append(puts)
        self.calls_df = pd.concat(all_calls, ignore_index=True)
        self.puts_df  = pd.concat(all_puts,  ignore_index=True)
        return self.calls_df, self.puts_df

    def fetch_iv_df(self, min_days=7/365, max_days=6/12, strike_width=0.1):
        self.calls_df['expiry'] = pd.to_datetime(self.calls_df['expiry']).dt.tz_localize('UTC')
        self.puts_df['expiry']  = pd.to_datetime(self.puts_df['expiry']).dt.tz_localize('UTC')
        today = datetime.now(timezone.utc)
        self.calls_df['T'] = (self.calls_df['expiry'] - today).dt.days / 365
        self.puts_df['T']  = (self.puts_df['expiry']  - today).dt.days / 365

        calls_df = self.calls_df[(self.calls_df['T'] >= min_days) & (self.calls_df['T'] <= max_days)]
        puts_df  = self.puts_df[(self.puts_df['T']  >= min_days) & (self.puts_df['T']  <= max_days)]

        spot = self.fetch_spot_price()
        r = self.fetch_risk_free_rate()

        otm_calls = calls_df[calls_df['strike'] >= spot * 0.99].copy()
        otm_puts = puts_df[puts_df['strike']   <= spot * 1.01].copy()

        otm_calls = otm_calls[(otm_calls['strike'] >= (1 - strike_width) * spot) & (otm_calls['strike'] <= (1 + strike_width) * spot)].copy()
        otm_puts = otm_puts[(otm_puts['strike']  >= (1 - strike_width) * spot) & (otm_puts['strike']  <= (1 + strike_width) * spot)].copy()

        # Dynamic risk-free rate for log moneyness
        otm_calls['log_moneyness'] = np.log(otm_calls['strike'] / (spot * np.exp(r * otm_calls['T'])))
        otm_puts['log_moneyness']  = np.log(otm_puts['strike']  / (spot * np.exp(r * otm_puts['T'])))

        otm_calls = otm_calls[(otm_calls['impliedVolatility'] > 0.05) & (otm_calls['impliedVolatility'] < 0.6)]
        otm_puts = otm_puts[(otm_puts['impliedVolatility']  > 0.05) & (otm_puts['impliedVolatility']  < 0.6)]

        otm_calls['total_variance'] = (otm_calls['impliedVolatility'] ** 2) * otm_calls['T']
        otm_puts['total_variance'] = (otm_puts['impliedVolatility']  ** 2) * otm_puts['T']

        combined_df  = pd.concat([otm_calls[['log_moneyness', 'total_variance', 'T']], otm_puts[['log_moneyness',  'total_variance', 'T']]], ignore_index=True)
        self.iv_df = combined_df
        return combined_df

    def _check_butterfly(self, a, b, rho, m, sigma):
        """
        Butterfly arbitrage condition: b * (1 + |rho|) < 2 / sigma
        Returns True if no violation.
        """
        return b * (1 + abs(rho)) < 2 / sigma

    def calibrate_svi_smile(self, expiry):
        temp = self.iv_df[np.isclose(self.iv_df['T'], expiry)]
        temp = temp.sort_values('log_moneyness')
        k = temp['log_moneyness'].values
        w_mkt = temp['total_variance'].values

        if len(k) < 5:
            return None

        def svi_raw(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        def residuals(params, k, w):
            return svi_raw(k, *params) - w

        a0 = np.mean(w_mkt)
        x0 = [a0, 0.05, -0.5, np.median(k), 0.2]
        bounds = ([0.000, 1e-6, -0.999, -1.0, 1e-6], [0.5,   1.0,   0.999,  1.0,  2.0])
        res = least_squares(residuals, x0, args=(k, w_mkt), bounds=bounds, method='trf')
        a, b, rho, m, sigma = res.x

        butterfly_ok = self._check_butterfly(a, b, rho, m, sigma)
        if not butterfly_ok:
            print(f"Butterfly violation at T={expiry:.3f} — b*(1+|rho|)={b*(1+abs(rho)):.4f} >= 2/sigma={2/sigma:.4f}")

        k_grid = np.linspace(k.min(), k.max(), 200)
        w_fit = svi_raw(k_grid, a, b, rho, m, sigma)
        return {'k': k, 'w_mkt': w_mkt, 'w_fit': w_fit, 'k_grid': k_grid, 'params': (a, b, rho, m, sigma), 'T': expiry, 'butterfly_ok': butterfly_ok}

    def plot_multiple_smiles(self):
        expiries = sorted(self.iv_df['T'].unique())
        fitted_smiles = {}
        for expiry in expiries:
            print(f"\nFitting T = {expiry:.3f}...")
            result = self.calibrate_svi_smile(expiry)
            if result is None:
                print(f"  Skipping — insufficient data")
                continue
            fitted_smiles[expiry] = result
            a, b, rho, m, sigma = result['params']
            print(f"  a={a:.6f}, b={b:.6f}, rho={rho:.4f}, m={m:.4f}, sigma={sigma:.4f}  butterfly={'✅' if result['butterfly_ok'] else '❌'}")
        return fitted_smiles

    def plot_smiles(self, fitted_smiles, num_smiles=8):
        expiries = sorted(fitted_smiles.keys())
        if len(expiries) > num_smiles:
            indices = np.linspace(0, len(expiries) - 1, num_smiles, dtype=int)
            expiries = [expiries[i] for i in indices]

        n_cols = 4
        n_rows = (len(expiries) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for i, expiry in enumerate(expiries):
            result = fitted_smiles[expiry]
            days = int(expiry * 365)
            ax = axes[i]
            ax.set_facecolor('#1e1e1e')
            ax.scatter(result['k'], result['w_mkt'], color='cyan', s=20, alpha=0.7, label='Market', zorder=3)
            ax.plot(result['k_grid'], result['w_fit'], color='lime', lw=2, label='SVI Fit')
            ax.set_title(f'T = {expiry:.3f}  ({days}d)', color='white', fontsize=10)
            ax.set_xlabel('Log Moneyness', color='#aaaaaa', fontsize=8)
            ax.set_ylabel('Total Variance', color='#aaaaaa', fontsize=8)
            ax.tick_params(colors='white', labelsize=7)
            ax.legend(facecolor='#2d2d2d', edgecolor='#555555', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.2, color='#555555')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.patch.set_facecolor('#1e1e1e')
        plt.suptitle(f'{self.ticker} SVI Smile Fits', color='white', fontsize=13)
        plt.tight_layout()
        plt.show()

    def build_surface_grid(self, fitted_smiles, num_k=50, num_T=50):
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

        a_interp = PchipInterpolator(expiries, a_list)
        b_interp = PchipInterpolator(expiries, b_list)
        rho_interp = PchipInterpolator(expiries, rho_list)
        m_interp = PchipInterpolator(expiries, m_list)
        sigma_interp = PchipInterpolator(expiries, sigma_list)

        k_grid = np.linspace(k_min, k_max, num_k)
        T_grid = np.linspace(min(expiries), max(expiries), num_T)
        IV_grid = np.zeros((num_T, num_k))   # (num_T, num_k) — rows=maturities, cols=strikes

        def svi_total_variance(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        for j, T in enumerate(T_grid):
            a, b, rho, m, sigma = (float(a_interp(T)), float(b_interp(T)), float(rho_interp(T)), float(m_interp(T)), float(sigma_interp(T)))
            w = svi_total_variance(k_grid, a, b, rho, m, sigma)
            w = np.maximum(w, 1e-8)        # floor at zero to avoid sqrt of negative
            IV_grid[j, :] = np.sqrt(w / T)

        # Enforce calendar arbitrage: total variance monotone in T
        w_grid = IV_grid ** 2 * T_grid[:, np.newaxis]  # back to total variance
        for j in range(1, num_T):
            w_grid[j, :] = np.maximum(w_grid[j, :], w_grid[j-1, :])
        IV_grid = np.sqrt(w_grid / T_grid[:, np.newaxis])
        return k_grid, T_grid, IV_grid



if __name__ == "__main__":
    svi = SviSurface(ticker='^SPX')
    svi.fetch_option_chain()
    svi.fetch_iv_df(min_days=15/365, max_days=6/12, strike_width=0.2)

    fitted_smiles = svi.plot_multiple_smiles()
    svi.plot_smiles(fitted_smiles, num_smiles=8)
    k_grid, T_grid, IV_grid = svi.build_surface_grid(fitted_smiles, num_k=50)

    # meshgrid
    K, T = np.meshgrid(k_grid, T_grid)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, T, IV_grid, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Log Moneyness')
    ax.set_ylabel('Maturity (Years)')
    ax.set_zlabel('Implied Vol')
    ax.set_title(f'{svi.ticker} SVI Implied Vol Surface')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Implied Vol')
    plt.show()