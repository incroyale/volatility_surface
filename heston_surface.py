import matplotlib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from svi_surface import *

plt.style.use('dark_background')
matplotlib.use("Qt5Agg")


class HestonSurface(SviSurface):

    def __init__(self, ticker="^SPX"):
        super().__init__(ticker)
        self.params= None


    def heston_cf(self, u, T, r, params):
        """
        Heston characteristic function φ(u)

        Parameters:
        -----------
        u : complex or np.array of complex
        T : float (maturity)
        r : float (risk-free rate)
        params : tuple (kappa, theta, sigma, rho, v0)

        Returns:
        --------
        complex value φ(u)
        """
        kappa, theta, sigma, rho, v0 = params
        i = 1j
        b = kappa - rho * sigma * i * u
        d = np.sqrt(b ** 2 + sigma ** 2 * (u ** 2 + i * u))
        g = (b - d) / (b + d) # Little Heston Trap
        exp_dT = np.exp(-d * T) # Avoid Division Explosion
        C = (i * u * r * T + (kappa * theta / sigma ** 2) * ((b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))))
        D = ((b - d) / sigma ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))
        phi = np.exp(C + D * v0 + i * u * np.log(self.spot_price))
        return phi



if __name__ == "__main__":
    heston = HestonSurface(ticker="^SPX")
    heston.spot_price = heston.fetch_spot_price()
    params = (2.0, 0.04, 0.5, -0.7, 0.04)  # kappa, theta, sigma, rho, v0

    T = 0.5
    r = 0.03

    # Check φ(0) = 1
    phi_0 = heston.heston_cf(0.0, T, r, params)
    print("phi(0) =", phi_0)
    u_vals = np.linspace(0.01, 50, 500)
    cf_vals = heston.heston_cf(u_vals, T, r, params)

    print("Any NaNs?", np.any(np.isnan(cf_vals)))
    print("Any Infs?", np.any(np.isinf(cf_vals)))

    # Plot real & imaginary parts
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(u_vals, np.real(cf_vals), label="Real part")
    plt.plot(u_vals, np.imag(cf_vals), label="Imag part")
    plt.legend()
    plt.title("Heston Characteristic Function")
    plt.show()






