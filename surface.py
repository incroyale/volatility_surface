import matplotlib
matplotlib.use("Qt5Agg")
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
plt.style.use('dark_background')

# Left wing of the surface (strikes below spot) → use OTM puts
# Right wing of the surface (strikes above spot) → use OTM calls


class volSurface:

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
            calls = chain.calls[['strike', 'bid', 'ask','impliedVolatility', 'inTheMoney']]
            puts = chain.puts[['strike', 'bid', 'ask','impliedVolatility', 'inTheMoney']]
            calls['mid'] = 0.5 * (calls['bid'] + calls['ask'])
            puts['mid'] = 0.5 * (puts['bid'] + puts['ask'])
            # calls['type'] = "C"
            # puts['type'] = "P"
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            all_calls.append(calls)
            all_puts.append(puts)

        self.calls_df = pd.concat(all_calls, ignore_index=True)
        self.puts_df = pd.concat(all_puts, ignore_index=True)
        return self.calls_df, self.puts_df


    def fetch_iv_df(self, min_days=0, max_days=0.5, strike_width=0.2):
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

        # Keep only OTM options
        otm_calls = calls_df[calls_df['inTheMoney'] == False]
        otm_puts = puts_df[puts_df['inTheMoney'] == False]

        # Filter Strike
        spot = self.fetch_spot_price()
        otm_calls = otm_calls[(otm_calls['strike'] >= (1 - strike_width) * spot) & (otm_calls['strike'] <= (1 + strike_width) * spot)]
        otm_puts = otm_puts[(otm_puts['strike'] >= (1 - strike_width) * spot) & (otm_puts['strike'] <= (1 + strike_width) * spot)]
        otm_calls = otm_calls.sort_values('strike')
        otm_puts = otm_puts.sort_values('strike')
        otm_calls['impliedVolatility'] = otm_calls['impliedVolatility'].interpolate(method='linear', limit_direction='both')
        otm_puts['impliedVolatility'] = otm_puts['impliedVolatility'].interpolate(method='linear', limit_direction='both')
        otm_calls = otm_calls.dropna(subset=['impliedVolatility'])
        otm_puts = otm_puts.dropna(subset=['impliedVolatility'])

        # IV Cleanup
        otm_calls = otm_calls[(otm_calls['impliedVolatility'] > 0.10) & (otm_calls['impliedVolatility'] < 0.6)]
        otm_puts = otm_puts[(otm_puts['impliedVolatility'] > 0.10) & (otm_puts['impliedVolatility'] < 0.6)]
        combined_df = pd.concat([otm_calls[['strike', 'impliedVolatility', 'T']],
                                 otm_puts[['strike', 'impliedVolatility', 'T']]], ignore_index=True)
        combined_df['total_variance'] = (combined_df['impliedVolatility'] ** 2) * combined_df['T']
        self.iv_df = combined_df
        return combined_df


    def interpolate_smile(self, term):
        """
        Interpolates the volatility smile using Hermite interpolation.
        Returns the ATM IV and the interpolated smile DataFrame.
        """
        # Get data for specific term
        smile_df = self.iv_df[self.iv_df['T'] == term].copy()
        smile_df = smile_df.sort_values("strike")
        strikes = smile_df["strike"].values
        ivs = smile_df["impliedVolatility"].values

        # Hermite interpolation
        interp = PchipInterpolator(strikes, ivs)
        spot = self.fetch_spot_price()
        atm_iv = float(interp(spot))

        # Dense grid for smooth plotting
        strike_grid = np.linspace(strikes.min(), strikes.max(), 200)
        iv_interpolated = interp(strike_grid)
        interpolated_df = pd.DataFrame({'strike': strike_grid,'impliedVolatility': iv_interpolated,'T': term})
        return atm_iv, interpolated_df, smile_df

    def get_available_strikes(self):
        return sorted(self.iv_df['strike'].unique())

    def get_available_terms(self):
        return sorted(self.iv_df['T'].unique())

    def get_term_structure(self, strike):
        term_df = self.iv_df[self.iv_df['strike'] == strike].sort_values('T')
        return term_df


    def plot_smile(self, term):
        """
        Plots the volatility smile with Hermite interpolation and marks ATM.
        """
        atm_iv, interpolated_df, original_df = self.interpolate_smile(term)
        spot = self.fetch_spot_price()
        plt.figure(figsize=(8, 5))
        plt.scatter(original_df["strike"], original_df["impliedVolatility"], color='cyan', s=50, label='Market Data', zorder=3)
        plt.plot(interpolated_df["strike"], interpolated_df["impliedVolatility"], color='cyan', linewidth=2, label='Hermite Interpolation')
        plt.axvline(spot, color='red', linestyle='--', alpha=0.5, label=f'ATM (Spot={spot:.0f})') # Mark ATM point
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.title(f"Implied Volatility Smile (T={term:.4f})")
        plt.grid(True, alpha=0.3)
        plt.show()
        return atm_iv


    def plot_term_structure(self, strike):
        term_df = self.get_term_structure(strike)
        plt.figure(figsize=(6, 4))
        plt.plot(term_df["T"], term_df["impliedVolatility"], marker="o")
        plt.xlabel("Time to Maturity (T)")
        plt.ylabel("Implied Volatility")
        plt.title("Implied Volatility vs. Term")
        plt.grid(True)
        plt.show()


    def build_iv_surface(self):
        unique_terms = sorted(self.iv_df['T'].unique())
        strikes = self.iv_df['strike'].values
        strike_range = np.linspace(strikes.min(), strikes.max(), 100)
        interpolated_smiles = []
        for term in unique_terms:
            smile_df = self.iv_df[self.iv_df['T'] == term].copy()
            smile_df = smile_df.sort_values("strike")
            strikes_term = smile_df["strike"].values
            ivs_term = smile_df["impliedVolatility"].values
            interp = PchipInterpolator(strikes_term, ivs_term)
            iv_interpolated = interp(strike_range)
            iv_interpolated = np.clip(iv_interpolated, 0.1, 0.5)
            interpolated_smiles.append(iv_interpolated)
        iv_matrix = np.array(interpolated_smiles)
        T_range = np.array(unique_terms)
        strike_grid = np.tile(strike_range, (len(T_range), 1))
        T_grid = np.tile(T_range.reshape(-1, 1), (1, len(strike_range)))
        iv_surface_initial = iv_matrix
        iv_surface = np.zeros_like(iv_surface_initial)
        for i in range(len(strike_range)):
            valid_mask = ~np.isnan(iv_surface_initial[:, i])
            if valid_mask.sum() > 1:
                interp_time = PchipInterpolator(T_range[valid_mask], iv_surface_initial[valid_mask, i])
                iv_surface[:, i] = interp_time(T_range)
            else:
                iv_surface[:, i] = iv_surface_initial[:, i]
        iv_surface = np.clip(iv_surface, 0.1, 0.5)
        return strike_grid, T_grid, iv_surface

    def plot_iv_surface_3d(self):
        strike_grid, T_grid, iv_surface = self.build_iv_surface()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(strike_grid, T_grid, iv_surface, cmap='inferno', alpha=1, edgecolor='none', antialiased=True)
        ax.set_xlabel('Strike', fontsize=10)
        ax.set_ylabel('Time to Maturity (Years)', fontsize=10)
        ax.set_zlabel('Implied Volatility', fontsize=10)
        ax.set_title('Implied Volatility Surface', fontsize=14)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.view_init(elev=25, azim=225)
        plt.tight_layout()
        plt.show()


def main():
    vs = volSurface(ticker="^SPX")
    vs.fetch_option_chain()
    vs.fetch_iv_df(strike_width=0.1, max_days=0.3, min_days=0)
    strike_grid, T_grid, iv_surface = vs.build_iv_surface()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(strike_grid, T_grid, iv_surface, cmap='inferno', alpha=1, edgecolor='none',antialiased=True)
    ax.set_xlabel('Strike', fontsize=10)
    ax.set_ylabel('Time to Maturity (Years)', fontsize=10)
    ax.set_zlabel('Implied Volatility', fontsize=10)
    ax.set_title('Implied Volatility Surface', fontsize=14)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=25, azim=225)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()










