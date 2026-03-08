# Extension of SVI - (Surface SVI)

from models.svi import SVI
import numpy as np
import pandas as pd
import matplotlib
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
plt.style.use('dark_background')



class SSVI(SVI):

    # Step 2: Define SSVI equation
    def total_variance(self, k, theta_t, rho, eta, gamma):
        phi = eta / (theta_t ** gamma * (1 + theta_t) ** (1 - gamma))
        w = (theta_t / 2) * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))
        return w

    def calibrate_ssvi(self):
        # Step 1: Extract Total ATM Variance using Linear Interpolation for log moneyness = 0
        thetas = {}
        for expiry, group in self.iv_df.groupby('T'):
            group = group.sort_values(by=['log_moneyness'])
            theta = np.interp(0, group['log_moneyness'], group['total_variance'])
            thetas[expiry] = theta

        # Step 3: Objective Function optimization
        def objective(params, thetas):
            rho, eta, gamma = params
            total_error = 0

            for T, group in self.iv_df.groupby('T'):
                theta_t = thetas[T]
                k = group['log_moneyness'].values
                w_market = group['total_variance'].values
                # Equal Weighting Used for now
                w_model = self.total_variance(k, theta_t, rho, eta, gamma)
                total_error += np.sum((w_model - w_market)**2)
            return total_error

        # Step 4: Parameter Bounds and arbitrage free Constraints
        bounds = [(-0.999, 0.999), # rho
                  (1e-6, 10.0), # eta
                  (1e-6, 1.0)] # gamma

        # Arbitrage Conditions
        def calendar_constraint(params, thetas):
            """ eta * (1 + |rho|) <= 2 """
            rho, eta, gamma = params
            return 2 - eta * (1 + abs(rho))

        def butterfly_constraint(params, thetas):
            """
            theta * phi^2 / 4 * (1 + |rho|)^2 <= 1 for all T
            i.e. min over all T of: 1 - theta * phi^2 / 4 * (1 + |rho|)^2 >= 0
            """
            rho, eta, gamma = params
            violations = []
            for theta_t in thetas.values():
                phi = eta / (theta_t ** gamma * (1 + theta_t) ** (1 - gamma))
                val = 1 - (theta_t * phi**2 / 4) * (1 + abs(rho))**2
                violations.append(val)
            return min(violations)

        constraints = [{'type': 'ineq', 'fun': calendar_constraint, 'args': (thetas,)}, {'type': 'ineq', 'fun': butterfly_constraint, 'args': (thetas, )}]

        # Step 5: Minimization
        def minimization():
            count = [0]
            # Phase 1: Global Search with differential evolution
            def obj_de(params):
                count[0] += 1
                error = objective(params, thetas)
                print(f"{count[0]} | error: {error:.6e}")
                return error

            bounds_de = [(-0.999, 0.999), (1e-6, 10.0), (1e-6, 1.0)] # rho, eta, gamma
            de_result = differential_evolution(obj_de, bounds=bounds_de, seed=42, maxiter=1000, tol=-1e-8, polish=False)

            # Phase 2: Local refinement from the best global point
            constraints = [{'type': 'ineq', 'fun': calendar_constraint, 'args': (thetas,)}, {'type': 'ineq', 'fun': butterfly_constraint, 'args': (thetas,)}]
            bounds_local = [(-0.999, 0.999), (1e-6, 10.0), (1e-6, 1.0)]

            local_result = minimize(objective, x0=de_result.x, args=(thetas, ),method='SLSQP',
                                    bounds=bounds_local, constraints=constraints, options={'ftol':1e-12, 'maxiter':1000})
            rho, eta, gamma = local_result.x
            return rho, eta, gamma, local_result

        self.thetas = thetas
        self.rho, self.eta, self.gamma, result = minimization()
        print(f"rho: {self.rho:.4f}")
        print(f"eta: {self.eta:.4f}")
        print(f"gamma: {self.gamma:.4f}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Final error: {result.fun:.6e}")

    def plot_ssvi_fit(self, num_maturities=9):
        expiries = sorted(self.iv_df['T'].unique())
        if len(expiries) > num_maturities:
            indices = np.linspace(0, len(expiries) - 1, num_maturities, dtype=int)
            expiries = [expiries[i] for i in indices]

        n_cols = 3
        n_rows = (len(expiries) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, T in enumerate(expiries):
            group = self.iv_df[self.iv_df['T'] == T].sort_values('log_moneyness')
            k = group['log_moneyness'].values
            w_market = group['total_variance'].values
            theta_t = self.thetas[T]

            k_grid = np.linspace(k.min(), k.max(), 200)
            w_model = self.total_variance(k_grid, theta_t, self.rho, self.eta, self.gamma)

            days = int(T * 365)
            axes[i].scatter(k, w_market, label='Market', s=25, alpha=0.6, color='cyan')
            axes[i].plot(k_grid, w_model, label='SSVI Fit', linewidth=2, color='lime')
            axes[i].set_title(f'T={T:.4f} ({days} days)')
            axes[i].set_xlabel('Log Moneyness')
            axes[i].set_ylabel('Total Variance')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        for j in range(len(expiries), len(axes)):
            axes[j].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.show()

    def build_ssvi_surface(self, num_k=50, num_T=50):
        """Build surface directly from global SSVI params — no per-slice fitting."""
        k_min = self.iv_df['log_moneyness'].min()
        k_max = self.iv_df['log_moneyness'].max()
        T_min = self.iv_df['T'].min()
        T_max = self.iv_df['T'].max()

        k_grid = np.linspace(k_min, k_max, num_k)
        T_grid = np.linspace(T_min, T_max, num_T)
        IV_grid = np.zeros((num_T, num_k))

        for j, T in enumerate(T_grid):
            theta_t = np.interp(T, sorted(self.thetas.keys()),
                                [self.thetas[t] for t in sorted(self.thetas.keys())])
            w = self.total_variance(k_grid, theta_t, self.rho, self.eta, self.gamma)
            w = np.clip(w, 1e-8, None)
            IV_grid[j, :] = np.sqrt(w / T)

        return k_grid, T_grid, IV_grid


if __name__ == '__main__':
    ssvi = SSVI()
    ssvi.iv_df = pd.read_csv("../data/sample_data.csv")
    print("Loading stored data from sample_data.csv")
    ssvi.calibrate_ssvi()
    ssvi.plot_ssvi_fit()
    k_grid, T_grid, IV_grid = ssvi.build_ssvi_surface(num_k=50, num_T=50)
    ssvi.plot_3d_surface(k_grid, T_grid, IV_grid)
