import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from svi_surface import SviSurface
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
import matplotlib

matplotlib.use("Qt5Agg")
plt.style.use('dark_background')


class HestonSurface(SviSurface):

    def __init__(self, ticker="^SPX", r=0.035, q=0.012):
        super().__init__(ticker=ticker)
        self.r = r
        self.q = q

    def _ql_setup(self):
        today = datetime.now(timezone.utc)
        calc_date = ql.Date(today.day, today.month, today.year)
        ql.Settings.instance().evaluationDate = calc_date
        dc = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, self.r, dc))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, self.q, dc))
        return calc_date, calendar, dc, flat_ts, div_ts

    def heston_iv_ql(self, K_arr, v0, kappa, theta, sigma, rho, tau):
        calc_date, calendar, dc, flat_ts, div_ts = self._ql_setup()
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))
        process = ql.HestonProcess(flat_ts, div_ts, spot_handle, v0, kappa, theta, sigma, rho)
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
        t_days = max(1, int(round(tau * 365)))
        period = ql.Period(t_days, ql.Days)
        iv_out = []

        for K in K_arr:
            try:
                helper = ql.HestonModelHelper(period, calendar, self.spot_price, float(K),ql.QuoteHandle(ql.SimpleQuote(0.3)),flat_ts, div_ts)
                helper.setPricingEngine(engine)
                iv_out.append(helper.impliedVolatility(helper.modelValue(), 1e-6, 1000, 0.001, 5.0))
            except Exception:
                iv_out.append(np.nan)
        return np.array(iv_out)

    def fetch_iv_df(self, min_days=2/365, max_days=1/12, strike_width=0.1):
        self.calls_df['expiry'] = pd.to_datetime(self.calls_df['expiry']).dt.tz_localize('UTC')
        self.puts_df['expiry']  = pd.to_datetime(self.puts_df['expiry']).dt.tz_localize('UTC')
        today = datetime.now(timezone.utc)
        self.calls_df['T'] = (self.calls_df['expiry'] - today).dt.days / 365
        self.puts_df['T']  = (self.puts_df['expiry']  - today).dt.days / 365

        calls_df = self.calls_df[(self.calls_df['T'] >= min_days) & (self.calls_df['T'] <= max_days)]
        puts_df  = self.puts_df[ (self.puts_df['T']  >= min_days) & (self.puts_df['T']  <= max_days)]
        spot     = self.fetch_spot_price()

        otm_calls = calls_df[calls_df['strike'] >= spot * 0.99]
        otm_puts  = puts_df[ puts_df['strike']  <= spot * 1.01]

        otm_calls = otm_calls[(otm_calls['strike'] >= (1 - strike_width) * spot) & (otm_calls['strike'] <= (1 + strike_width) * spot)]
        otm_puts  = otm_puts[ (otm_puts['strike']  >= (1 - strike_width) * spot) & (otm_puts['strike']  <= (1 + strike_width) * spot)]

        otm_calls = otm_calls[(otm_calls['impliedVolatility'] > 0.05) & (otm_calls['impliedVolatility'] < 0.80)]
        otm_puts  = otm_puts[ (otm_puts['impliedVolatility']  > 0.05) & (otm_puts['impliedVolatility']  < 0.80)]

        combined_df = pd.concat([otm_calls[['strike', 'impliedVolatility', 'T']], otm_puts[[ 'strike', 'impliedVolatility', 'T']]], ignore_index=True)
        self.iv_df = (combined_df.groupby(['T', 'strike'], as_index=False)['impliedVolatility'].mean())
        return self.iv_df

    def calibrate_heston_smile(self, expiry):
        temp = self.iv_df[np.isclose(self.iv_df['T'], expiry, atol=1e-6)].sort_values('strike')
        if len(temp) < 3:
            raise ValueError(f"Too few strikes ({len(temp)}) for T={expiry:.4f}")

        K_mkt  = temp['strike'].values
        iv_mkt = temp['impliedVolatility'].values

        def residuals(params):
            kappa, theta, v0, sigma, rho = params
            iv_model = self.heston_iv_ql(K_mkt, v0, kappa, theta, sigma, rho, expiry)
            return np.where(np.isnan(iv_model), 1.0, iv_model) - iv_mkt

        v0_guess = float(np.median(iv_mkt)**2)
        x0 = [2.0,  v0_guess, v0_guess, 0.4,   -0.7]
        lb = [0.01, 0.001, 0.001, 0.001, -0.999]
        ub = [20.0, 2.0, 2.0, 2.0, 0.999]

        res = least_squares(residuals, x0, bounds=(lb, ub), method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=2000)
        kappa, theta, v0, sigma, rho = res.x
        return {'params': (kappa, theta, v0, sigma, rho), 'T': expiry}

    def calibrate_all_smiles(self):
        expiries = sorted(self.iv_df['T'].unique())
        fitted_smiles = {}
        for expiry in expiries:
            print(f"Calibrating T = {expiry:.4f} yrs ({int(round(expiry * 365))} days)...")
            try:
                result = self.calibrate_heston_smile(expiry)
                kappa, theta, v0, sigma, rho = result['params']
                print(f"  kappa={kappa:.4f}  theta={theta:.4f}  v0={v0:.4f}  sigma={sigma:.4f}  rho={rho:.4f}")
                fitted_smiles[expiry] = result
            except Exception as e:
                print(f"  Skipped — {e}")
        return fitted_smiles

    def build_surface_grid(self, fitted_smiles, num_k=60, num_T=60):
        expiries = sorted(fitted_smiles.keys())
        kappa_l, theta_l, v0_l, sigma_l, rho_l = [], [], [], [], []
        K_min, K_max = np.inf, -np.inf

        for T in expiries:
            kappa, theta, v0, sigma, rho = fitted_smiles[T]['params']
            kappa_l.append(kappa); theta_l.append(theta); v0_l.append(v0)
            sigma_l.append(sigma); rho_l.append(rho)
            K_min = min(K_min, self.iv_df[np.isclose(self.iv_df['T'], T, atol=1e-6)]['strike'].min())
            K_max = max(K_max, self.iv_df[np.isclose(self.iv_df['T'], T, atol=1e-6)]['strike'].max())

        kappa_interp = PchipInterpolator(expiries, kappa_l)
        theta_interp = PchipInterpolator(expiries, theta_l)
        v0_interp    = PchipInterpolator(expiries, v0_l)
        sigma_interp = PchipInterpolator(expiries, sigma_l)
        rho_interp   = PchipInterpolator(expiries, rho_l)

        K_grid  = np.linspace(K_min, K_max, num_k)
        T_grid  = np.linspace(min(expiries), max(expiries), num_T)
        IV_grid = np.full((num_k, num_T), np.nan)

        for j, T in enumerate(T_grid):
            kappa = float(np.clip(kappa_interp(T), 0.01, 20.0))
            theta = float(np.clip(theta_interp(T), 0.001, 2.0))
            v0    = float(np.clip(v0_interp(T),    0.001, 2.0))
            sigma = float(np.clip(sigma_interp(T), 0.001, 2.0))
            rho   = float(np.clip(rho_interp(T),  -0.999, 0.999))
            IV_grid[:, j] = self.heston_iv_ql(K_grid, v0, kappa, theta, sigma, rho, T)

        return K_grid, T_grid, IV_grid

    def plot_surface(self, K_grid, T_grid, IV_grid):
        K, T = np.meshgrid(K_grid, T_grid, indexing='ij')
        fig  = plt.figure(figsize=(12, 7))
        ax   = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K, T, IV_grid * 100, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.87)
        ax.set_xlabel('Strike (K)')
        ax.set_ylabel('Maturity (Years)')
        ax.set_zlabel('Implied Vol (%)')
        ax.set_title(f'{self.ticker} — Heston Implied Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Implied Vol (%)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    heston = HestonSurface(ticker="^SPX", r=0.035, q=0.012)
    heston.fetch_option_chain()
    heston.spot_price = heston.fetch_spot_price()
    print(f"Spot: {heston.spot_price:.2f}")
    heston.fetch_iv_df(min_days=3/365, max_days=10/365, strike_width=0.12)
    fitted_smiles = heston.calibrate_all_smiles()
    K_grid, T_grid, IV_grid = heston.build_surface_grid(fitted_smiles)
    heston.plot_surface(K_grid, T_grid, IV_grid)
