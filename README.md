# Implied Volatility Surface SPX Options

Calibration of implied volatility surfaces from live SPX options data using the **SVI (Stochastic Volatility Inspired)**, **SSVI (Surface SVI)**, and **Heston** stochastic volatility models. The project covers the full pipeline: data acquisition, OTM filtering, calibration, and 3D surface visualisation.

---

## Data Pipeline

Options data is fetched live from Yahoo Finance via `yfinance`. For each expiry in the option chain, OTM calls and puts are selected and merged into a unified DataFrame.

**Filtering rules applied:**
- Expiry range: configurable `[min_days, max_days]` window (e.g. 2 days to 2 months)
- Strike range: within ±10% of spot by default
- OTM selection: calls with `K >= 0.99 * S`, puts with `K <= 1.01 * S`
- IV cleanup: `0.10 < IV < 0.50` to strip stale or illiquid quotes

For SVI and SSVI, strikes are converted to **log-moneyness** `k = ln(K/F)` where `F = S * exp(r * T)` is the forward price, and **total variance** `w = IV^2 * T` is used as the fitting target. This is the natural space for SVI and makes no-arbitrage conditions easier to enforce.

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

## Model 2: SSVI (Surface SVI)

### Parametrisation

SSVI extends SVI by directly modelling the entire surface with a single globally consistent parametrisation. Total variance is:

```
w(k, theta) = (theta / 2) * ( 1 + rho * phi(theta) * k + sqrt((phi(theta) * k + rho)^2 + (1 - rho^2)) )
```

where the power-law wings function is:

```
phi(theta) = eta / ( theta^gamma * (1 + theta)^(1 - gamma) )
```

and `theta(T)` is the ATM total variance at each maturity, extracted via linear interpolation.

**Parameter interpretation:**

| Parameter | Role |
|---|---|
| `rho` in `(-1, 1)` | Global skew |
| `eta > 0` | Overall vol-of-vol level |
| `gamma` in `(0, 1)` | Controls how wings decay with maturity |

- Unlike SVI, there is no per-slice fitting. `rho`, `eta`, and `gamma` are calibrated once across the entire surface simultaneously.
- Optimisation uses a two-phase approach: **differential evolution** for global search followed by **SLSQP** for local refinement.
- No-arbitrage is enforced via explicit constraints: calendar `eta * (1 + |rho|) <= 2` and butterfly `theta * phi^2 * (1 + |rho|)^2 <= 4`.

### Results
![SSVI Smile Fit Per Expiry](https://github.com/incroyale/volatility_surface/blob/main/images/ssvi_smiles.png "SSVI Per-Expiry Smile Fits")

![SSVI IV Surface](https://github.com/incroyale/volatility_surface/blob/main/images/ssvi_surface.png "SSVI Implied Volatility")
