import numpy as np
import matplotlib.pyplot as plt
import os

# Load Data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

#Load problem data
returns = np.load(os.path.join(data_dir, 'expected_returns.npy'))
cov_matrix = np.load(os.path.join(data_dir, 'covariance_matrix.npy'))
sectors = np.load(os.path.join(data_dir, 'sector_list.npy'), allow_pickle=True)
volatilities = np.load(os.path.join(data_dir, 'volatilities.npy'))
classical_selected = np.load(os.path.join(data_dir, 'classical_selected.npy'))
classical_weights = np.load(os.path.join(data_dir, 'classical_weights.npy'))
quantum_selected = np.load(os.path.join(data_dir, 'quantum_selected.npy'))
quantum_weights = np.load(os.path.join(data_dir, 'quantum_weights.npy'))


def calc_portfolio_metrics(selected, weights, returns, cov_matrix, rf=0.02):
    """Calculate portfolio return, risk, and Sharpe ratio."""
    selected = list(selected)
    port_return = np.dot(weights, returns[selected])
    port_risk = np.sqrt(weights @ cov_matrix[np.ix_(selected, selected)] @ weights)
    sharpe = (port_return - rf) / port_risk
    return port_return, port_risk, sharpe

# Classical metrics 
c_return, c_risk, c_sharpe = calc_portfolio_metrics(
    classical_selected, classical_weights, returns, cov_matrix
)
# Quantum metrics
q_return, q_risk, q_sharpe = calc_portfolio_metrics(
    quantum_selected, quantum_weights, returns, cov_matrix
)

# Set Style
plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Create output directory for figures
fig_dir= os.path.join(base_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Portfolio Performance comparison', fontsize=14, fontweight='bold')

# Return comparison
ax1 = axes[0]
methods = ['Classical/Baseline', 'Quantum Solution']
returns_vals = [c_return * 100, q_return * 100]
colors = ['#3498db', '#e74c3c']
bars1 = ax1.bar(methods, returns_vals, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Expected Return (%)')
ax1.set_title('Expected_Return')
for bar, val in zip(bars1, returns_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold')

# Risk Comparison
ax2 = axes[1]
risk_vals = [c_risk * 100, q_risk * 100]
bars2 = ax2.bar(methods, risk_vals, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Risk / Volatility (%)')
ax2.set_title('Portfolio Risk')
for bar, val in zip(bars2, risk_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')

#Sharpe comparsion
ax3 = axes[2]
sharpe_vals = [c_sharpe, q_sharpe]
bars3 = ax3.bar(methods, sharpe_vals, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Sharpe Ratio')
ax3.set_title('sharpe ratio (Higher = Better)')
for bar, val in zip(bars3, sharpe_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', fontweight='bold')
    
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'performance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Sector Allocation Pie Chart

# Count sectors in quantum portfolio
sector_counts = {}
for i in quantum_selected:
    s = str(sectors[i])
    sector_counts[s] = sector_counts.get(s, 0) + 1

fig, ax = plt.subplots(figsize=(8, 6))
sector_names = list(sector_counts.keys())
sector_values = list(sector_counts.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(sector_names)))

wedges, texts, autotexts = ax.pie(
    sector_values, labels=sector_names, autopct='%1.1f%%',
    colors=colors, startangle=90, explode=[0.02] * len(sector_names)
)
for autotext in autotexts:
    autotext.set_fontweight('bold')

ax.set_title('Portfolio Sector Allocation (Quantum Solution)', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(fig_dir, 'sector_allocation.png'), dpi=150, bbox_inches='tight')
plt.close()

# Asset Weights Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

#Sort by weight
sorted_pairs = sorted(zip(quantum_weights, quantum_selected), reverse=True)
weights_sorted = [w * 100 for w, _ in sorted_pairs]
assets_sorted = [f"Asset {a}" for _, a in sorted_pairs]

colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(assets_sorted)))[::-1]
bars = ax.barh(assets_sorted[::-1], weights_sorted[::-1], color=colors, edgecolor='black')

ax.set_xlabel('Weight (%)', fontsize=12)
ax.set_title('Optimal Portfolio Weights ( Quantum Solution)', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(weights_sorted) * 1.2)

# Add value labels
for bar, val in zip(bars, weights_sorted[::-1]):
    if val > 1:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'asset_weights.png'), dpi=150, bbox_inches='tight')
plt.close()

#  Risk-Return Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot all assets
asset_vols = volatilities * 100
asset_rets = returns * 100

ax.scatter(asset_vols, asset_rets, c='lightgray', s=50, alpha=0.6, label='All Assets')

# Highlight selected assets
selected_vols = asset_vols[quantum_selected]
selected_rets = asset_rets[quantum_selected]
ax.scatter(selected_vols, selected_rets, c='#2ecc71', s=100, edgecolor='black', 
           linewidth=1.5, label='Selected (Quantum)', zorder=5)

# Plot portfolio
ax.scatter([q_risk * 100], [q_return * 100], c='#e74c3c', s=300, marker='*',
           edgecolor='black', linewidth=2, label='Quantum Portfolio', zorder=10)

ax.set_xlabel('Risk / Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Risk-Return Analysis', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'risk_return_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# Innovation Summary Infographic
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Quantum Portfolio Optimization - Innovations', 
        fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)

# Innovation 1: Hybrid Warm Start
ax.text(0.1, 0.75, 'Innovation 1: Hybrid Warm Start', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.68, '• Uses classical heuristics to initialize quantum annealing', fontsize=10,
        transform=ax.transAxes)
ax.text(0.1, 0.62, '• 6 warm start methods: Greedy Sharpe, Favor Holdings, Sector Balanced, etc.', fontsize=10,
        transform=ax.transAxes)
ax.text(0.1, 0.56, '• Result: Faster convergence, better solution quality', fontsize=10,
        transform=ax.transAxes)

# Innovation 2: Rebalancing Trigger
ax.text(0.1, 0.42, 'Innovation 2: Rebalancing Trigger', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.35, '• Uses Mahalanobis distance to measure portfolio drift', fontsize=10,
        transform=ax.transAxes)
ax.text(0.1, 0.29, '• Accounts for correlations between assets', fontsize=10,
        transform=ax.transAxes)
ax.text(0.1, 0.23, '• Result: Smart rebalancing, reduced transaction costs', fontsize=10,
        transform=ax.transAxes)

# Results box
ax.add_patch(plt.Rectangle((0.55, 0.15), 0.4, 0.55, fill=True, 
             facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2,
             transform=ax.transAxes))

ax.text(0.75, 0.62, 'KEY RESULTS', fontsize=12, fontweight='bold', ha='center',
        transform=ax.transAxes)
ax.text(0.75, 0.52, f'Sharpe Ratio: {q_sharpe:.4f}', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.75, 0.44, f'Return: {q_return*100:.2f}%', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.75, 0.36, f'Risk: {q_risk*100:.2f}%', fontsize=11, ha='center',
        transform=ax.transAxes)
ax.text(0.75, 0.26, f'Assets: {len(quantum_selected)}', fontsize=11, ha='center',
        transform=ax.transAxes)

plt.savefig(os.path.join(fig_dir, 'innovation_summary.png'), dpi=150, bbox_inches='tight')
plt.close()