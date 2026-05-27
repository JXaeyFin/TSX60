# MPT Data-Driven Asset Allocation Model

A quantitative research project implementing **Modern Portfolio Theory (MPT)** on TSX 60 constituents, using SLSQP optimization to construct and out-of-sample test two portfolios against Canadian market benchmarks.

**Author:** Jeffrey Xia  
**Data:** CHASS/CFMRC TSX 60 daily closing prices (2020–2024) · yfinance (2026 YTD)  
**Risk-Free Rate:** 2.68% (Canadian 1-Year Treasury)

---

## Results

The key finding is that **training window length significantly impacts out-of-sample performance**. Shorter, more recent windows consistently outperform longer ones — suggesting recent price action in the TSX 60 carries more predictive signal than older data.

| Training Window | Max Sharpe (Portfolio) | Max Sharpe (TSX 60) |
|:-:|:-:|:-:|
| 0.5 Year | **1.189** | 0.422 |
| 1 Year   | **1.049** | 0.422 |
| 2 Year   | **0.547** | 0.422 |
| 3 Year   | 0.288 | 0.422 |
| 4 Year   | 0.236 | 0.422 |
| 5 Year   | **0.476** | 0.422 |

Best result (0.5-year window): **13.98% YTD return, 9.50% volatility, 1.189 Sharpe** vs TSX 60 benchmark at 8.66% return, 14.18% volatility, 0.422 Sharpe.

A Jobson-Korkie significance test applied to the 126-day out-of-sample return series yields a z-statistic of 8.44 (p ≈ 0). While statistically significant over this window, this result should be treated as promising evidence rather than proof — the optimization implicitly fits to historical patterns that may not persist, and a single short window is insufficient to rule out a favorable market regime. Replication across multiple non-overlapping periods would be required to make a stronger claim.

---

## Methodology

### 1. Data Pipeline 
- 77,338 datapoints loaded from CHASS/CFMRC database (TSX 60 daily closing prices)
- AI tools leveraged for code development: VSC Copilot, Gemini 3.1 Pro, ChatGPT Codex, Claude Code.
- Long-format CSV pivoted to wide returns matrix via pandas
- Tickers with incomplete history dropped to ensure full-rank covariance matrix

### 2. Statistical Inputs
- **Expected returns (μ):** Mean daily return × 252 (annualized)
- **Covariance matrix (Σ):** Daily covariance × 252 (annualized)
- **Equilibrium assumption:** Historical returns proxy future returns

### 3. Optimization (SLSQP)
Two portfolios are constructed using Sequential Least Squares Quadratic Programming:

- **Max Sharpe Portfolio** — maximizes `(wᵀμ - rf) / √(wᵀΣw)` by minimizing its negative
- **Min Volatility Portfolio** — minimizes `√(wᵀΣw)` independent of expected returns

Constraints: weights sum to 1 (fully invested), no short selling (weights ≥ 0).

SLSQP uses quadratic approximation via second-order Taylor expansion to iteratively solve the constrained nonlinear problem. It is appropriate here because the min-volatility problem is strictly convex and the max-Sharpe problem is approximately convex for well-behaved covariance matrices.

### 4. Out-of-Sample Testing
Optimized weights are applied forward to YTD 2026 live prices (via yfinance) without reoptimization. Performance is benchmarked against:
- **XIU.TO** — iShares S&P/TSX 60 Index ETF (float-adjusted market cap)
- **^GSPTSE** — TSX Composite (220 stocks, float-adjusted market cap)

### 5. Statistical Significance
Jobson-Korkie (1981) test applied to realized out-of-sample Sharpe ratio differences. See code documentation in Block 8 for full derivation and limitation discussion.

---

## Usage

### Requirements
```
numpy
pandas
scipy
matplotlib
yfinance
```

Install via:
```bash
pip install numpy pandas scipy matplotlib yfinance
```

### Running the Model
1. Export TSX 60 daily returns from CHASS/CFMRC as `cfmrc_data.csv` and place it in the project directory
2. Adjust `training_years` in Block 1 to set the lookback window (e.g. `0.5`, `1`, `2`)
3. Run:
```bash
python MPT.py
```

Output: console performance tables + `portfolio_vs_markets_ytd.png`

---

## Limitations

- **Equilibrium assumption:** Historical returns are used as proxies for future returns. This assumption breaks down during structural market regime changes.
- **Equity-only universe:** The efficiency frontier is constrained to public equities. Including bonds, REITs, and alternative assets would expand the frontier and likely improve risk-adjusted performance.
- **Single out-of-sample window:** Statistical significance over one 126-day period cannot distinguish a genuine edge from a favorable market regime.
- **No transaction costs or rebalancing:** Live implementation would incur trading costs and require a rebalancing schedule, both of which would reduce realized returns.
- **Survivorship bias:** Tickers unavailable via yfinance are dropped and weights renormalized, introducing minor survivorship bias.

---

## Next Steps

1. Expand asset universe to bonds, REITs, and alternative funds
2. Test recency hypothesis on a different time period (e.g. 2014–2020)
3. Implement Black-Litterman model to relax the equilibrium assumption
4. Automate sentiment inputs via FinBERT on SEC/SEDAR financial reporting data
5. Apply Memmel (2003) correction to Jobson-Korkie test for small-sample robustness
6. Test across multiple non-overlapping out-of-sample windows to validate significance
