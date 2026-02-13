import numpy as np
from scipy.integrate import quad
from svi_surface import SviSurface # check svi_surface.py file
from datetime import datetime, timezone
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


class HestonSurface(SviSurface):

    def __init__(self, ticker="^SPX", r=0.035, lambd=0.0):
        super().__init__(ticker=ticker)
        self.r = r
        self.lambd = lambd


    @staticmethod
    def char_func(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
        a = kappa * theta
        b = kappa + lambd
        rspi = rho * sigma * phi * 1j
        d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)
        g = (b - rspi + d) / (b - rspi - d)

        # calculate characteristic function by components
        exp1  = np.exp(r * phi * 1j * tau)
        term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
        exp2  = np.exp(a * tau * (b - rspi + d) / sigma**2 + v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2)
        return exp1 * term2 * exp2

    @staticmethod
    def integrand(phi, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        numerator   = np.exp(r * tau) * HestonSurface.char_func(phi - 1j, *args) - K * HestonSurface.char_func(phi, *args)
        denominator = 1j * phi * K**(1j * phi)
        return numerator / denominator

    @staticmethod
    def call_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
        args = (S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
        real_integral, err = np.real(quad(HestonSurface.integrand, 0, 100, args=args))
        return (S0 - K * np.exp(-r * tau)) / 2 + real_integral / np.pi


    def fetch_iv_df(self, min_days=2/365, max_days=1/12, strike_width=0.1):
        """
        Overriding method from svi_surface
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

        # IV Cleanup
        otm_calls = otm_calls[(otm_calls['impliedVolatility'] > 0.10) & (otm_calls['impliedVolatility'] < 0.5)]
        otm_puts = otm_puts[(otm_puts['impliedVolatility'] > 0.10) & (otm_puts['impliedVolatility'] < 0.5)]

        combined_df = pd.concat([otm_calls[['strike', 'impliedVolatility', 'T']], otm_puts[['strike', 'impliedVolatility', 'T']]], ignore_index=True)
        self.iv_df = combined_df
        return combined_df

    def bs_invert(self, price, S0, K, T, r):
        def bs_call(sigma):
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
        if price <= intrinsic:
            return np.nan

        try:
            return brentq(lambda sigma: bs_call(sigma) - price, 1e-6, 10.0)
        except (ValueError, RuntimeError):
            return np.nan

    def calibrate_heston_smile(self, expiry):
        # 1. Slice iv_df for this expiry
        temp = self.iv_df[self.iv_df['T'] == expiry].sort_values('strike')
        K_mkt = temp['strike'].values
        iv_mkt = temp['impliedVolatility'].values
        S0 = self.spot_price
        T = expiry
        r = self.r

        # 2. Helper — converts an array of strikes to model IVs
        def heston_iv_vec(K_arr, kappa, theta, v0, sigma, rho):
            iv_model = []
            for K in K_arr:
                price = self.call_price(S0, K, v0, kappa, theta, sigma, rho, self.lambd, T, r)
                iv = self.bs_invert(price, S0, K, T, r)
                iv_model.append(iv)
            return np.array(iv_model)

        # 3. Residuals
        def residuals(params):
            kappa, theta, v0, sigma, rho = params
            iv_model = heston_iv_vec(K_mkt, kappa, theta, v0, sigma, rho)
            # replace NaNs with large penalty instead of dropping them
            iv_model = np.where(np.isnan(iv_model), 1.0, iv_model)
            return iv_model - iv_mkt

        # 4. Initial Guess and Bounds
        v0_guess = float(np.median(iv_mkt) ** 2)  # rough guess: ATM variance
        x0 = [2.0, v0_guess, v0_guess, 0.4, -0.6]
        #     kappa    theta     v0        sigma  rho
        lb = [0.01, 0.001, 0.001, 0.001, -0.999]
        ub = [20.0, 2.0, 2.0, 2.0, 0.999]

        # 5. Optimiser
        res = least_squares(residuals, x0, bounds=(lb, ub), method='trf')
        kappa, theta, v0, sigma, rho = res.x
        return {'params': (kappa, theta, v0, sigma, rho), 'T': expiry, 'K_mkt': K_mkt, 'iv_mkt': iv_mkt}

    def calibrate_all_smiles(self):
        expiries = sorted(self.iv_df['T'].unique())
        fitted_smiles = {}

        for expiry in expiries:
            print(f"Calibrating T = {expiry:.4f} yrs ({int(round(expiry * 365))} days)...")
            try:
                result = self.calibrate_heston_smile(expiry)
                fitted_smiles[expiry] = result
            except Exception as e:
                print(f"  Skipping — {e}")

        return fitted_smiles

    def build_surface_grid(self, fitted_smiles, num_k=50, num_T=50):
        expiries = sorted(fitted_smiles.keys())

        kappa_l, theta_l, v0_l, sigma_l, rho_l = [], [], [], [], []
        K_min, K_max = np.inf, -np.inf

        for T in expiries:
            kappa, theta, v0, sigma, rho = fitted_smiles[T]['params']
            kappa_l.append(kappa)
            theta_l.append(theta)
            v0_l.append(v0)
            sigma_l.append(sigma)
            rho_l.append(rho)
            K_min = min(K_min, fitted_smiles[T]['K_mkt'].min())
            K_max = max(K_max, fitted_smiles[T]['K_mkt'].max())

        # PCHIP interpolate each param across maturities
        kappa_interp = PchipInterpolator(expiries, kappa_l)
        theta_interp = PchipInterpolator(expiries, theta_l)
        v0_interp = PchipInterpolator(expiries, v0_l)
        sigma_interp = PchipInterpolator(expiries, sigma_l)
        rho_interp = PchipInterpolator(expiries, rho_l)

        K_grid = np.linspace(K_min, K_max, num_k)
        T_grid = np.linspace(min(expiries), max(expiries), num_T)
        IV_grid = np.full((num_k, num_T), np.nan)

        for j, T in enumerate(T_grid):
            kappa = float(kappa_interp(T))
            theta = float(theta_interp(T))
            v0 = float(v0_interp(T))
            sigma = float(sigma_interp(T))
            rho = float(rho_interp(T))
            for i, K in enumerate(K_grid):
                price = self.call_price(self.spot_price, K, v0, kappa, theta, sigma, rho, self.lambd, T, self.r)
                iv = self.bs_invert(price, self.spot_price, K, T, self.r)
                IV_grid[i, j] = iv
        return K_grid, T_grid, IV_grid


    def plot_surface(self, K_grid, T_grid, IV_grid):
        K, T = np.meshgrid(K_grid, T_grid, indexing='ij')
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K, T, IV_grid, cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.85)
        ax.set_xlabel('Strike (K)')
        ax.set_ylabel('Maturity (Years)')
        ax.set_zlabel('Implied Vol')
        ax.set_title(f'{self.ticker} — Heston Implied Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Implied Vol')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    heston = HestonSurface(ticker="^SPX")
    heston.fetch_option_chain()
    heston.spot_price = heston.fetch_spot_price()
    heston.fetch_iv_df(min_days=10/365, max_days=11/365)
    fitted_smiles = heston.calibrate_all_smiles()
    K_grid, T_grid, IV_grid = heston.build_surface_grid(fitted_smiles)
    heston.plot_surface(K_grid, T_grid, IV_grid)
