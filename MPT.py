"""
================================================================================
MPT Data-Driven Asset Allocation Model
================================================================================
Author:      Jeffrey Xia
Tools Used: VSCode, VSC Copilot, Gemini 3.1 Pro, ChatGPT Codex, Claude Code
Description: Implements Modern Portfolio Theory (MPT) on TSX 60 constituents
             using SLSQP optimization to construct two portfolios:
               1. Maximum Sharpe Ratio (Tangency Portfolio)
               2. Minimum Volatility (Global Minimum Variance Portfolio)

             Both portfolios are evaluated out-of-sample against the TSX 60
             Index (XIU.TO) and TSX Composite (^GSPTSE) benchmarks using
             YTD 2026 live market data.

             A Jobson-Korkie significance test is applied to assess whether
             out-of-sample Sharpe ratio outperformance is statistically
             meaningful.

Methodology: SLSQP via scipy.optimize.minimize
Data:        CHASS/CFMRC daily closing prices (TSX 60), 2020-2024
             yfinance live prices for out-of-sample testing (2026 YTD)
Risk-Free:   2.68% (Canadian 1-Year Treasury, as of model construction)

Dependencies: numpy, pandas, scipy, matplotlib, yfinance
================================================================================
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple

# =============================================================================
# BLOCK 1: DATA INGESTION AND RESHAPING
# =============================================================================

print("Loading CFMRC data...")
try:
    df = pd.read_csv('cfmrc_data.csv')
except FileNotFoundError:
    print("\nERROR: 'cfmrc_data.csv' not found.")
    print("Please ensure your CFMRC export file is in the same folder as this script.")
    sys.exit()

df = df.rename(columns={
    'trdate-Trade Date': 'Date',
    'symbol-Ticker': 'Ticker',
    'return-Daily Return': 'Return',
    'ind24-S&P/TSX 60 Daily Total Return Index': 'TSX60_Index'
})

df['Date'] = pd.to_datetime(df['Date'])
df = df.drop_duplicates(subset=['Date', 'Ticker'], keep='first')

# ── TRAINING WINDOW CONFIGURATION ────────────────────────────────────────────
training_years = 0.5 
training_end_date = pd.Timestamp('2025-12-31')
training_months = int(round(training_years * 12))
training_start_date = (
    training_end_date
    - pd.DateOffset(months=training_months)
    + pd.Timedelta(days=1)
)

df = df[(df['Date'] >= training_start_date) & (df['Date'] <= training_end_date)]
print(f"Using training data from {training_start_date.date()} to {training_end_date.date()} ({training_years} years).")

print("Pivoting data to wide format...")
returns_matrix = df.pivot(index='Date', columns='Ticker', values='Return')
returns_matrix = returns_matrix.dropna(axis=1)


# =============================================================================
# BLOCK 2: STATISTICAL FOUNDATIONS
# =============================================================================

columns_to_drop = [col for col in returns_matrix.columns if pd.isna(col) or col == 'XIU']
clean_matrix = returns_matrix.drop(columns=columns_to_drop, errors='ignore')

tickers = clean_matrix.columns.tolist()
num_assets = len(tickers)
print(f"Optimizing a universe of {num_assets} TSX assets...\n")

mu = clean_matrix.mean() * 252
Sigma = clean_matrix.cov() * 252
risk_free_rate = 0.0268  


# =============================================================================
# BLOCK 3: OBJECTIVE FUNCTIONS
# =============================================================================

def portfolio_performance(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float) -> Tuple[float, float, float]:
    """Computes annualized return, volatility, and Sharpe ratio."""
    weights = np.array(weights)
    returns = np.dot(weights, mu)
    risk = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe = (returns - risk_free_rate) / risk
    return returns, risk, sharpe


def negative_sharpe_ratio(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float) -> float:
    """Objective function for Sharpe maximization."""
    _, _, sharpe = portfolio_performance(weights, mu, Sigma, risk_free_rate)
    return -sharpe


def minimize_volatility(weights: np.ndarray, Sigma: pd.DataFrame) -> float:
    """Objective function for minimum variance optimization."""
    return np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))


# =============================================================================
# BLOCK 4: CONSTRAINTS AND BOUNDS
# =============================================================================

initial_guess = np.array(num_assets * [1. / num_assets])
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))


# =============================================================================
# BLOCK 5: SLSQP OPTIMIZATION
# =============================================================================

print("Running SLSQP Optimization engines...")

# Optimizer 1: Maximum Sharpe Ratio 
optimal_result_sharpe = minimize(
    fun=negative_sharpe_ratio,
    x0=initial_guess,
    args=(mu, Sigma, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
optimal_weights_sharpe = optimal_result_sharpe.x
opt_ret_S, opt_risk_S, opt_sharpe_S = portfolio_performance(optimal_weights_sharpe, mu, Sigma, risk_free_rate)

# Optimizer 2: Minimum Volatility 
optimal_result_vol = minimize(
    fun=minimize_volatility,
    x0=initial_guess,
    args=(Sigma,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
optimal_weights_vol = optimal_result_vol.x
opt_ret_V, opt_risk_V, opt_sharpe_V = portfolio_performance(optimal_weights_vol, mu, Sigma, risk_free_rate)


# =============================================================================
# BLOCK 6: IN-SAMPLE RESULTS
# =============================================================================

print("\n" + "=" * 60)
print(f"{'ASSET ALLOCATIONS (>0.01%)':^60}")
print("=" * 60)
print(f"{'Ticker':<10} | {'Max Sharpe Wgt':<20} | {'Min Vol Wgt':<20}")
print("-" * 60)

for i, ticker in enumerate(tickers):
    w_sharpe = optimal_weights_sharpe[i] * 100
    w_vol = optimal_weights_vol[i] * 100
    if w_sharpe > 0.01 or w_vol > 0.01:
        print(f"{ticker:<10} | {w_sharpe:>10.2f}%{' ':>9} | {w_vol:>9.2f}%")


# =============================================================================
# BLOCK 7: OUT-OF-SAMPLE BACKTESTING
# =============================================================================

print("\n" + "=" * 60)
print("OUT-OF-SAMPLE TESTING (YTD 2026)")
print("=" * 60)

yf_tickers = [f"{ticker.replace('.', '-')}.TO" for ticker in tickers]

print("Fetching live portfolio prices...")
test_data = yf.download(yf_tickers, start="2026-01-01", end="2026-07-01", progress=False)['Close']
print("Fetching Market Benchmarks (TSX 60 & TSX Composite)...")
benchmark_data = yf.download(["XIU.TO", "^GSPTSE"], start="2026-01-01", end="2026-07-01", progress=False)['Close']

test_returns = test_data.pct_change().dropna()
benchmark_returns = benchmark_data.pct_change().dropna()

cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
test_returns = test_returns[test_returns.index <= cutoff_date]
benchmark_returns = benchmark_returns[benchmark_returns.index <= cutoff_date]

# Survivorship check and weight renormalization
surviving_tickers = [t for t in yf_tickers if t in test_returns.columns]
surv_wgt_sharpe = np.array([optimal_weights_sharpe[i] for i, t in enumerate(yf_tickers) if t in surviving_tickers])
surv_wgt_vol = np.array([optimal_weights_vol[i] for i, t in enumerate(yf_tickers) if t in surviving_tickers])

if np.sum(surv_wgt_sharpe) > 0:
    surv_wgt_sharpe /= np.sum(surv_wgt_sharpe)
if np.sum(surv_wgt_vol) > 0:
    surv_wgt_vol /= np.sum(surv_wgt_vol)

# Ensure precise matrix alignment
test_returns = test_returns[surviving_tickers]

port_returns_sharpe = test_returns.dot(surv_wgt_sharpe)
port_returns_vol = test_returns.dot(surv_wgt_vol)

master_df = pd.DataFrame({
    'Max_Sharpe': port_returns_sharpe,
    'Min_Vol': port_returns_vol,
    'TSX_60': benchmark_returns['XIU.TO'],
    'TSX_Composite': benchmark_returns['^GSPTSE']
}).dropna()

cumulative_returns = (1 + master_df).cumprod() - 1

# Performance Metrics (Renamed to accurately reflect Total Return)
ytd_ret_sharpe = master_df['Max_Sharpe'].add(1).prod() - 1
ytd_ret_vol = master_df['Min_Vol'].add(1).prod() - 1
ytd_ret_60 = master_df['TSX_60'].add(1).prod() - 1
ytd_ret_comp = master_df['TSX_Composite'].add(1).prod() - 1

oos_vol_sharpe = master_df['Max_Sharpe'].std() * np.sqrt(252)
oos_vol_vol = master_df['Min_Vol'].std() * np.sqrt(252)
oos_vol_60 = master_df['TSX_60'].std() * np.sqrt(252)
oos_vol_comp = master_df['TSX_Composite'].std() * np.sqrt(252)

print("-" * 60)
print(f"Test Period: {master_df.index[0].strftime('%Y-%m-%d')} to {master_df.index[-1].strftime('%Y-%m-%d')}")
print("-" * 60)
print(f"{'Portfolio/Index':<25} | {'YTD Return':<12} | {'YTD Volatility'}")
print("-" * 60)
print(f"{'Max Sharpe Portfolio':<25} | {ytd_ret_sharpe * 100:>9.2f}%   | {oos_vol_sharpe * 100:>10.2f}%")
print(f"{'Min Vol Portfolio':<25} | {ytd_ret_vol * 100:>9.2f}%   | {oos_vol_vol * 100:>10.2f}%")
print(f"{'TSX 60 (Large Cap)':<25} | {ytd_ret_60 * 100:>9.2f}%   | {oos_vol_60 * 100:>10.2f}%")
print(f"{'TSX Composite (All)':<25} | {ytd_ret_comp * 100:>9.2f}%   | {oos_vol_comp * 100:>10.2f}%")
print("-" * 60)


# =============================================================================
# BLOCK 8: STATISTICAL SIGNIFICANCE TESTING (Jobson-Korkie)
# =============================================================================

def jobson_korkie_test(returns_a: pd.Series, returns_b: pd.Series, risk_free_rate: float, periods: int = 252) -> Tuple[float, float, float, float]:
    """Jobson-Korkie (1981) test for significance of Sharpe ratio difference."""
    rf_daily = risk_free_rate / periods
    excess_a, excess_b = returns_a - rf_daily, returns_b - rf_daily
    n = len(returns_a)

    sr_a = (excess_a.mean() / excess_a.std()) * np.sqrt(periods)
    sr_b = (excess_b.mean() / excess_b.std()) * np.sqrt(periods)
    rho = np.corrcoef(returns_a, returns_b)[0, 1]

    sigma_sq = (1 / n) * (2 - 2 * rho + 0.5 * sr_a ** 2 + 0.5 * sr_b ** 2 - sr_a * sr_b * rho ** 2)
    std_error = np.sqrt(sigma_sq)

    z_stat = (sr_a - sr_b) / std_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value, sr_a, sr_b


print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE (Jobson-Korkie 1981)")
print("=" * 60)

comparisons = [
    ("Max Sharpe vs TSX 60",        master_df['Max_Sharpe'], master_df['TSX_60']),
    ("Max Sharpe vs TSX Composite", master_df['Max_Sharpe'], master_df['TSX_Composite']),
    ("Min Vol vs TSX 60",           master_df['Min_Vol'],    master_df['TSX_60']),
    ("Min Vol vs TSX Composite",    master_df['Min_Vol'],    master_df['TSX_Composite']),
]

for name, ret_a, ret_b in comparisons:
    z, p, sr_a, sr_b = jobson_korkie_test(ret_a, ret_b, risk_free_rate)
    significance = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"\n  {name}")
    print(f"    Sharpe (portfolio): {sr_a:.3f}  |  Sharpe (benchmark): {sr_b:.3f}")
    print(f"    Z-statistic: {z:.3f}  |  P-value: {p:.4f}  ({significance} at 95%)")


# =============================================================================
# BLOCK 9: DATA VISUALIZATION
# =============================================================================

print("\nGenerating Equity Curve chart...")
fig, ax = plt.subplots(figsize=(14, 8))

dates = cumulative_returns.index
ret_sharpe = cumulative_returns['Max_Sharpe'] * 100
ret_vol = cumulative_returns['Min_Vol'] * 100
ret_tsx60 = cumulative_returns['TSX_60'] * 100
ret_comp = cumulative_returns['TSX_Composite'] * 100

color_sharpe, color_vol = '#1f77b4', '#d62728'
color_tsx60, color_comp = '#8c8c8c', '#bfbfbf'

ax.plot(dates, ret_sharpe, label='Max Sharpe Portfolio', color=color_sharpe, linestyle='-', linewidth=3, marker='o', markersize=4, markevery=5)
ax.plot(dates, ret_vol, label='Min Volatility Portfolio', color=color_vol, linestyle='-', linewidth=3, marker='D', markersize=4, markevery=5)
ax.plot(dates, ret_tsx60, label='TSX 60 (XIU.TO)', color=color_tsx60, linestyle=':', linewidth=1.5)
ax.plot(dates, ret_comp, label='TSX Composite (^GSPTSE)', color=color_comp, linestyle=':', linewidth=1.5)


def plot_global_extremes(series_dict: dict, colors_dict: dict):
    """Annotates absolute highs and lows with collision detection."""
    placed_texts = []
    min_x_days_clearance = 12
    min_y_clearance = 1.2

    points = []
    for label_name, series in series_dict.items():
        points.append({'date': series.idxmax(), 'val': series.max(), 'type': 'Max', 'color': colors_dict[label_name]})
        points.append({'date': series.idxmin(), 'val': series.min(), 'type': 'Min', 'color': colors_dict[label_name]})

    points.sort(key=lambda pt: pt['date'])

    for p in points:
        x_date, y_val, kind, color = p['date'], p['val'], p['type'], p['color']
        marker_style = '^' if kind == 'Max' else 'v'
        ax.plot(x_date, y_val, marker=marker_style, color=color, markersize=7, markeredgecolor='white', zorder=5)

        target_y = y_val + (1.2 if kind == 'Max' else -1.2)
        for _ in range(20):
            if any(abs((x_date - px).days) < min_x_days_clearance and abs(target_y - py) < min_y_clearance for px, py in placed_texts):
                target_y += (0.5 if kind == 'Max' else -0.5)
            else:
                break

        placed_texts.append((x_date, target_y))
        ax.plot([x_date, x_date], [y_val, target_y], color=color, linestyle=':', alpha=0.6, linewidth=1.2)
        ax.text(x_date, target_y, f"{kind}: {y_val:.1f}%", color=color, fontsize=8, ha='center', va='center',
                fontweight='bold', zorder=6, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color))


series_to_label = {'Max Sharpe': ret_sharpe, 'Min Vol': ret_vol, 'TSX 60': ret_tsx60, 'TSX Comp': ret_comp}
colors_mapping = {'Max Sharpe': color_sharpe, 'Min Vol': color_vol, 'TSX 60': color_tsx60, 'TSX Comp': color_comp}
plot_global_extremes(series_to_label, colors_mapping)

# ── ENDPOINT LABELS ───────────────────────────────────────────────────────────
endpoint_labels = [
    ('Max Sharpe Portfolio', ret_sharpe.iloc[-1], color_sharpe),
    ('Min Volatility Portfolio', ret_vol.iloc[-1], color_vol),
    ('TSX 60 (XIU.TO)', ret_tsx60.iloc[-1], color_tsx60),
    ('TSX Composite (^GSPTSE)', ret_comp.iloc[-1], color_comp)
]
placed_positions = []
min_vertical_clearance = 0.8

for label, value, color in endpoint_labels:
    target_y = value
    for _ in range(20):
        if any(abs(target_y - existing_y) < min_vertical_clearance for existing_y in placed_positions):
            target_y += min_vertical_clearance
        else:
            break
    placed_positions.append(target_y)
    va = 'bottom' if target_y >= value else 'top'
    ax.text(dates[-1], target_y, f" {value:.2f}%", va=va, fontsize=9, fontweight='bold', color=color)

# ── VOLATILITY SUMMARY BOX ────────────────────────────────────────────────────
volatility_text = (
    "Out-of-Sample Volatility (Annualized)\n"
    "─────────────────────────────────────\n"
    f"Max Sharpe:       {oos_vol_sharpe * 100:>5.2f}%\n"
    f"Min Volatility:   {oos_vol_vol * 100:>5.2f}%\n"
    f"TSX 60:           {oos_vol_60 * 100:>5.2f}%\n"
    f"TSX Composite:    {oos_vol_comp * 100:>5.2f}%"
)
ax.text(0.98, 0.04, volatility_text, transform=ax.transAxes, fontsize=10, family='monospace',
        ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da'))

# ── FINAL STYLING ─────────────────────────────────────────────────────────────
ax.set_title(f'Out-of-Sample Performance: Optimizers vs. Broader Markets (YTD 2026) - Training Window: {training_years} years', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)

plt.xticks(rotation=45, fontsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('portfolio_vs_markets_ytd.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'portfolio_vs_markets_ytd.png'.")
