# Volatility Surface Modeling for Index Options

## Overview

This project implements volatility surface construction for S&P 500 index options using progressively sophisticated techniques, from basic interpolation to parametric model fitting with arbitrage-free constraints.

## Project Phases

### Phase 1: Raw Interpolation Surface

**Objective:** Construct a baseline volatility surface using market-observable data without model assumptions.

**Methodology:**
- Extracted out-of-the-money (OTM) call and put option data from yfinance API
- Used raw inputs: strike prices (K), time to expiry (T), and implied volatilities (σ)
- Applied Hermite cubic interpolation across both strike and maturity dimensions
- Generated volatility smiles (σ vs. strike) and term structures (σ vs. time)

**Limitations:**
- No arbitrage constraints enforced
- Purely data-driven approach susceptible to market data noise
- No guarantee of smooth or financially meaningful surface

![Phase 1](https://github.com/incroyale/volatility_surface/blob/main/images/phase_1_surface.png "Phase 1")

### Phase 2: SVI Model with Surface Interpolation

**Objective:** Improve surface quality by fitting parametric models to individual smiles while maintaining flexibility across maturities.

**Methodology:**
- Implemented the Stochastic Volatility Inspired (SVI) model for each maturity slice
- Raw SVI parameterization: `w(k) = a + b[ρ(k - m) + √((k - m)² + σ²)]`
  - where `w` is total variance, `k` is log-moneyness, and `{a, b, ρ, m, σ}` are fitted parameters
- Calibrated SVI parameters independently for each expiry using least-squares optimization
- Applied Hermite interpolation to construct the full surface across maturities
- Incorporated arbitrage-free constraints following Gatheral & Jacquier (2013)

![Phase 2 Smiles](https://github.com/incroyale/volatility_surface/blob/main/images/svi_smiles.png "Phase 2 Smiles")
![Phase 2 Surface](https://github.com/incroyale/volatility_surface/blob/main/images/svi_surface_1.png "Phase 2 Surface")
![Phase 2 Surface](https://github.com/incroyale/volatility_surface/blob/main/images/svi_surface_2.png "Phase 2 Surface")

### Phase 3: Heston Stochastic Volatility Model *(In Progress)*

**Objective:** Implement a full stochastic volatility framework for consistent pricing and risk management.

**Planned Methodology:**
- Calibrate Heston model parameters to the observed volatility surface
- Leverage characteristic function-based pricing for European options
- Validate model fit against SVI surface from Phase 2

**Status:** Under development


## References
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- Gatheral, J., & Jacquier, A. (2013). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.





