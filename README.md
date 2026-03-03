# Implied Volatility Surface â SPX Options

Calibration of implied volatility surfaces from live SPX options data using the **SVI (Stochastic Volatility Inspired)** and **Heston** stochastic volatility models. The project covers the full pipeline: data acquisition, OTM filtering, per-smile calibration, cross-maturity interpolation, and 3D surface visualisation.

---

## Project Structure

```
.
├── svi_surface.py       # SVI model — base class
├── heston_surface.py    # Heston model — inherits from sviSurface
├── sample_data.csv      # Stored SPX option data for offline runs
└── images/
    ├── svi_smiles.png
    ├── svi_surface_1.png
    └── heston_surface.png
```

---

## Data Pipeline

Options data is fetched live from Yahoo Finance via `yfinance`. For each expiry in the option chain, OTM calls and puts are selected and merged into a unified DataFrame.

**Filtering rules applied:**
- Expiry range: configurable `[min_days, max_days]` window (e.g. 2 days to 2 months)
- Strike range: within ±10% of spot by default
- OTM selection: calls with `K >= 0.99 * S`, puts with `K <= 1.01 * S`
- IV cleanup: `0.10 < IV < 0.50` to strip stale or illiquid quotes

For SVI, strikes are converted to **log-moneyness** `k = ln(K/F)` where `F = S * exp(r * T)` is the forward price, and **total variance** `w = IV^2 * T` is used as the fitting target. This is the natural space for SVI and makes no-arbitrage conditions easier to enforce.

For Heston, raw strikes and market IVs are used directly, as the model prices in strike space via numerical integration.

---

## Model 1 â SVI (Stochastic Volatility Inspired)

### Parametrisation

The raw SVI model fits total variance as a function of log-moneyness:

```
w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )
```

**Parameter interpretation:**

| Parameter | Role |
|---|---|
| `a` | Overall variance level (vertical shift) |
| `b >= 0` | Slope / wings steepness |
| `rho` in `(-1, 1)` | Skew — left/right asymmetry |
| `m` | Horizontal shift (ATM location) |
| `sigma > 0` | Curvature / smile smoothness |

The formula has the correct asymptotic behaviour: linear in `|k|` for large moneyness, consistent with power-law tails observed in equity options markets.

### Calibration

Each expiry slice is calibrated independently. The optimisation minimises squared total-variance residuals:

```
min  sum_i [ w_model(k_i; params) - w_market(k_i) ]^2
```

Solver: **Trust-Region Reflective** (`scipy.optimize.least_squares`, `method=trf`).

Bounds enforced to prevent arbitrage and keep the formula well-defined:
- `a in [0.0, 0.5]`
- `b in [1e-6, 1.0]`
- `rho in (-0.999, 0.999)`
- `m in [-1.0, 1.0]`
- `sigma in [1e-6, 2.0]`

Initial guess: `a0 = mean(w_market)`, `b0 = 0.05`, `rho0 = -0.5`, `m0 = median(k)`, `sigma0 = 0.2`.

### Surface Construction

After per-slice calibration, each of the five SVI parameters is treated as a smooth function of maturity. **PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)** is used to interpolate each parameter curve across maturities. PCHIP is preferred over natural cubic splines because it preserves local monotonicity and avoids spurious oscillations between calibrated nodes — important when parameters like `rho` or `b` vary non-monotonically across expiries.

A dense `(k, T)` grid is then evaluated:
1. Interpolate `(a, b, rho, m, sigma)` at each `T` via PCHIP
2. Evaluate `w(k; params(T))` for each `(k, T)` pair
3. Clip `w` to `[1e-8, inf)` to prevent negative variance
4. Convert: `IV(k, T) = sqrt(w / T)`

### Visualisation

- **Per-smile plots**: market scatter vs SVI fit in `(log-moneyness, total variance)` space, one subplot per expiry
- **3D surface**: dark-theme Bloomberg-style surface with projected smile contours on the lateral walls, coloured by maturity / moneyness

**SVI Per-Expiry Smile Fits:** https://github.com/incroyale/volatility_surface/blob/main/images/svi_smiles.png

**SVI Implied Volatility Surface:** https://github.com/incroyale/volatility_surface/blob/main/images/svi_surface_1.png

---

## Model 2 â Heston Stochastic Volatility

### Model Dynamics

The Heston model specifies asset price and variance jointly:

```
dS =  mu * S dt + sqrt(v) * S dW1
dv = kappa * (theta - v) dt + sigma * sqrt(v) dW2
corr(dW1, dW2) = rho
```

**Parameter interpretation:**

| Parameter | Role |
|---|---|
| `v0` | Initial (spot) variance |
| `kappa` | Mean-reversion speed of variance |
| `theta` | Long-run mean variance |
| `sigma` | Vol-of-vol (volatility of variance) |
| `rho` | Spot-vol correlation — primary skew driver |

The Feller condition `2 * kappa * theta > sigma^2` ensures variance stays strictly positive, though this is not explicitly enforced during calibration (bounds are used instead).

### Pricing via Characteristic Function

Call prices are computed using the **Heston characteristic function** and direct numerical quadrature (`scipy.integrate.quad` over `[0, 100]`). Define:

```
b     = kappa + lambda
rspi  = rho * sigma * phi * i
d     = sqrt( (rspi - b)^2 + (i*phi + phi^2) * sigma^2 )
g     = (b - rspi + d) / (b - rspi - d)
```

The characteristic function is:

```
phi(u) = exp(r*i*u*T) * S0^(i*u) * ((1 - g*exp(d*T)) / (1 - g))^(-2a/sigma^2)
       * exp( a*T*(b - rspi + d)/sigma^2 + v0*(b - rspi + d)*((1 - exp(d*T)) / (1 - g*exp(d*T)))/sigma^2 )
```

where `a = kappa * theta`. The call price is recovered as:

```
C = (S0 - K*exp(-r*T)) / 2  +  (1/pi) * Re[ integral_0^inf integrand(phi) dphi ]
```

with the integrand combining evaluations of the characteristic function at `phi` and `phi - i`.

### IV Recovery â Black-Scholes Inversion

Since the Heston model produces call prices rather than IVs directly, each model price is inverted to implied volatility via **Brent\'s method** (`scipy.optimize.brentq`), solving:

```
brentq( BS_call(sigma) - C_heston = 0,  sigma in [1e-6, 10.0] )
```

Intrinsic value is checked first: if the Heston price is at or below intrinsic `max(S - K*exp(-rT), 0)`, the IV is set to `NaN` and excluded from the calibration residuals.

### Calibration

Per-slice calibration minimises squared IV error across OTM strikes:

```
min  sum_i [ IV_model(K_i; kappa, theta, v0, sigma, rho) - IV_market(K_i) ]^2
```

NaN model IVs (from failed Brent inversion) are replaced with a large penalty value of `1.0` to keep the residual vector well-defined throughout the optimisation.

Solver: **Trust-Region Reflective** (`least_squares`, `trf`).

Initial guess and bounds:

| Param | x0 | Lower | Upper |
|---|---|---|---|
| `kappa` | 2.0 | 0.01 | 20.0 |
| `theta` | `median(IV)^2` | 0.001 | 2.0 |
| `v0` | `median(IV)^2` | 0.001 | 2.0 |
| `sigma` | 0.4 | 0.001 | 2.0 |
| `rho` | -0.6 | -0.999 | 0.999 |

### Surface Construction

Identical interpolation strategy to SVI: each of the five Heston parameters is PCHIP-interpolated across calibrated maturities. The surface is built by re-pricing via the characteristic function on a dense `(K, T)` grid and inverting each price to IV via Brent.

**Heston Implied Volatility Surface:** https://github.com/incroyale/volatility_surface/blob/main/images/heston_surface.png

---

## Methodology Comparison

| Step | SVI | Heston |
|---|---|---|
| **Input space** | Log-moneyness / total variance | Strike / implied vol |
| **Pricing** | Closed-form algebraic formula | Numerical integration of char. function |
| **IV recovery** | Direct: `sqrt(w / T)` | Black-Scholes inversion via Brent |
| **Calibration target** | Total variance `w(k)` | Implied volatility `IV(K)` |
| **Optimiser** | TRF least squares | TRF least squares |
| **Surface interpolation** | PCHIP on `(a, b, rho, m, sigma)` | PCHIP on `(kappa, theta, v0, sigma, rho)` |
| **Speed** | Fast (algebraic per point) | Slow (quadrature + Brent per point) |
| **Structural arbitrage** | Bounds on params; no calendar arb yet | Bounds on params; no calendar arb yet |

---

## Stack

`Python 3` Â· `NumPy` Â· `SciPy` Â· `pandas` Â· `yfinance` Â· `Matplotlib`

---

## Roadmap

- [ ] SSVI and eSSVI parametrisations with joint surface calibration
- [ ] Calendar and butterfly arbitrage-free constraints (Gatheral & Jacquier)
- [ ] Carr-Madan FFT pricing for Heston (stub in `call_price_fft`)
- [ ] Rough volatility models (rBergomi)
- [ ] Greeks surface: delta, vega, vanna, volga
- [ ] Local volatility extraction via Dupire
