# Implied Volatility Surface SPX Options

Calibration of implied volatility surfaces from live SPX options data using the **SVI (Stochastic Volatility Inspired)** and **Heston** stochastic volatility models. The project covers the full pipeline: data acquisition, OTM filtering, per-smile calibration, cross-maturity interpolation, and 3D surface visualisation.

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

## Model 1: SVI (Stochastic Volatility Inspired)

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


- Each expiry slice is calibrated independently. The optimisation minimises squared total-variance residuals using non-linear least squares.


- Uses **PCHIP** (Piecewise Cubic Hermite Interpolation Polynomial)

### Results
![SVI Smile Fit Per Expiry](https://github.com/incroyale/volatility_surface/blob/main/images/svi_smiles.png "SVI Per-Expiry Smile Fits")

![SVI IV Surface](https://github.com/incroyale/volatility_surface/blob/main/images/svi_surface_1.png "SVI Implied Volatility Surface")

---

## Model 2: Heston Stochastic Volatility

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

- The Feller condition `2 * kappa * theta > sigma^2` ensures variance stays strictly positive.


- Call prices are computed using the **Heston characteristic function**


- IV is calculated from Heston price using **brentq**

### Results
![Heston IV Surface](https://github.com/incroyale/volatility_surface/blob/main/images/heston_surface.png "Heston Implied Volatility Surface")
---
