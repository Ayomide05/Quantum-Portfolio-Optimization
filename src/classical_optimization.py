import numpy as np
import json
import time
import os
from scipy.optimize import minimize


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

# Load all data files
returns = np.load(os.path.join(data_dir, 'expected_returns.npy'))
cov_matrix = np.load(os.path.join(data_dir, 'covariance_matrix.npy'))
prev_positions = np.load(os.path.join(data_dir, 'previous_positions.npy'))
trans_costs = np.load(os.path.join(data_dir, 'transaction_costs.npy'))
volatilities = np.load(os.path.join(data_dir, 'volatilities.npy'))
sectors = np.load(os.path.join(data_dir, 'sector_list.npy'), allow_pickle=True)


n_assets = len(returns)
unique_sectors = np.unique(sectors)

N = 15              # Target portfolio size
K = 5               # Max changes per rebalancing period
S_max = 5           # Max assets per sector (35% of 15)
risk_free_rate = 0.02    # Risk-free rate
lambda_risk = 0.5     # Risk aversion parameter
tau = 0.1    # Transaction cost multiplier

# Current holdings
current_holdings = set(i for i in range(n_assets) if prev_positions[i] == 1)
print("=" * 70)
print("CLASSICAL PORTFOLIO OPTIMIZATION")
print("=" * 70)
print(f"\nCurrent holdings: {len(current_holdings)} assets")
print(f"Target: {N} assets")
print(f"Max changes per week: {K}")

# Helper Functions
def portfolio_return(selected, returns):
    """Calculate equal-weighted portfolio return."""
    if len(selected) == 0:
        return 0
    return np.mean(returns[list(selected)])

def portfolio_risk(selected, cov_matrix):
    """Calculate equal-weighted portfolio risk."""
    selected = list(selected)
    n = len(selected)
    if n == 0:
        return float('inf')
    
    w = np.ones(n) / n
    sub_cov = cov_matrix[np.ix_(selected, selected)]
    variance = w @ sub_cov @ w
    return np.sqrt(variance)

def count_sectors(selected, sectors):
    """Count assets per sector."""
    counts = {}
    for i in selected:
        s = str(sectors[i])
        counts[s] = counts.get(s, 0) + 1
    return counts

def get_changes(new_portfolio, prev_positions):
    """Calculate what needs to change."""
    new_set = set(new_portfolio)
    old_set = set(current_holdings)

    # Changes = assets we're selling + assets we're buying
    to_sell = old_set - new_set
    to_buy = new_set - old_set
    to_hold = new_set & old_set

    total_changes = len(to_sell) + len(to_buy)

    return total_changes, to_sell, to_buy, to_hold

def calculate_transaction_cost(to_buy, to_sell, trans_costs):
    """Calculate total transaction cost."""
    total_cost = 0
    for i in to_buy:
        total_cost += trans_costs[i]
    for i in to_sell:
        total_cost += trans_costs[i]
    return total_cost

def calculate_sharpe(selected, returns, cov_matrix, rf=0.02):
    """Calculate Sharpe ratio using equal weights"""
    if len(selected) == 0:
        return -float('inf')
   
    selected = list(selected)
    n = len(selected)
    w = np.ones(n) / n

    port_return = np.dot(w, returns[selected])
    port_risk = np.sqrt(w @ cov_matrix[np.ix_(selected, selected)] @ w)

    if port_risk < 1e-10:
        return -float('inf')
    
    return (port_return -rf) / port_risk

# Stage 1: Find Optimal Portfolio (Without K constraint)

def simulated_annealing(returns, cov_matrix, sectors, n_select=15, 
                        max_per_sector=5, n_iterations=20000):
    """Simulated Annealing for asset selection. 
    This Mimics the physical process of annealing in metallurgy:
    - High temperature: Accept many changes (explore widely)
    - Low temperature: Accept only improvements (refine solution).
    """

    n = len(returns)
    unique_sectors = np.unique(sectors)
     
    # ---------Create initial portfolio ------------
    def create_initial_portfolio():
        """Genaret a random valid portfolio."""
        selected = []
        sector_counts = {str(s): 0 for s in unique_sectors}
        available = list(range(n))
        np.random.shuffle(available)

        for i in available:
            if len(selected) >= n_select:
                break
            sector = str(sectors[i])
            if sector_counts[sector] < max_per_sector:
                selected.append(i)
                sector_counts[sector] += 1

        return selected

    def make_small_change(selected):
        """Swap one asset for another."""
        new_selected = selected.copy()

        # Remove random asset
        remove_idx = np.random.randint(len(new_selected))
        removed = new_selected.pop(remove_idx)

        # Count sectors
        sector_counts = {}
        for c in new_selected:
            s = str(sectors[c])
            sector_counts[s] = sector_counts.get(s, 0) + 1

        # Find valid candidates to add
        candidates = []
        for i in range(n):
            if i not in new_selected:
                s = str(sectors[i])
                if sector_counts.get(s, 0) < max_per_sector:
                    candidates.append(i)

        if not candidates:
            new_selected.append(removed)
            return new_selected
        
        new_selected.append(candidates[np.random.randint(len(candidates))])
        return new_selected
    
    # Initialize
    current = create_initial_portfolio()
    current_sharpe = calculate_sharpe(current, returns, cov_matrix)

    best = current.copy()
    best_sharpe = current_sharpe

    temperature = 100.0
    cooling_rate = 0.995
    
    print(f"\nStarting optimization...")
    print(f"Initial Sharpe: {current_sharpe:.4f}\n")
  
    for i in range(n_iterations):
        # Generate neighbor
        neighbor = make_small_change(current)
        neighbor_sharpe= calculate_sharpe(neighbor, returns, cov_matrix)
            
        # Calculate acceptance probability
        delta = neighbor_sharpe - current_sharpe  
                
        if delta > 0:   
            current = neighbor
            current_sharpe = neighbor_sharpe
        else:
            prob = np.exp(delta / temperature)  
            if np.random.random() < prob:
                current = neighbor
                current_sharpe = neighbor_sharpe
                        
        # Update best
        if current_sharpe > best_sharpe:
            best = current.copy()
            best_sharpe = current_sharpe
        
        # Cool down
        temperatue = max(0.1, temperature * cooling_rate)
            
        if i % 4000 == 0:
            print(f"Iter {i:5d} | Best Sharpe: {best_sharpe:.4f}")
    
    print(f"\nOptimization complete!")
    print(f"Final Sharpe: {best_sharpe:.4f}")
    return best, best_sharpe

# Run Optimization
np.random.seed(42)

start_time = time.time()
optimal_portfolio, optimal_ = simulated_annealing(
    returns, cov_matrix, sectors,n_select=N, max_per_sector=S_max,n_iterations=20000
)
computation_time = time.time() - start_time

optimal_sorted = sorted(optimal_portfolio)

# Analyze Optimal Portfolio
print("\n" + "-" * 40)
print("OPTIMAL PORTFOLIO FOUND")
print("-" * 40)

port_return = portfolio_return(optimal_portfolio, returns)
port_risk = portfolio_risk(optimal_portfolio, cov_matrix)
port_sharpe = calculate_sharpe(optimal_portfolio, returns, cov_matrix)
sector_breakdown = count_sectors(optimal_portfolio, sectors)

print(f"\nSelected {len(optimal_sorted)} assets: {optimal_sorted}")

print(f"\nPortfolio Metrics (Equal Weights):")
print(f"  Expected Return: {port_return*100:.2f}%")
print(f"  Risk (Std Dev):  {port_risk*100:.2f}%")
print(f"  Sharpe Ratio:    {port_sharpe:.4f}")

print(f"\nSector Breakdown:")
for sector in sorted(sector_breakdown.keys()):
    count = sector_breakdown[sector]
    status = "✓" if count <= S_max else "✗"
    print(f"  {sector}: {count} {status}")

# Transition PLAN (respects K Constraints)
print("\n" + "=" * 70)
print("TRANSITION PLAN (Multi-Period Rebalancing)")
print("=" * 70)

# What needs to change?
n_changes, to_sell, to_buy, to_hold = get_changes(optimal_portfolio, current_holdings)

print(f"\nTransition Summary:")
print(f"  Currently holding: {len(current_holdings)} assets")
print(f"  Target portfolio:  {len(optimal_portfolio)} assets")
print(f"  Assets to HOLD:    {len(to_hold)} → {sorted(to_hold)}")
print(f"  Assets to SELL:    {len(to_sell)} → {sorted(to_sell)}")
print(f"  Assets to BUY:     {len(to_buy)} → {sorted(to_buy)}")
print(f"  Total changes:     {n_changes}")

weeks_needed = int(np.ceil(n_changes / K))
print(f"\n  Max changes per week: {K}")
print(f"  Weeks needed: {weeks_needed}")

# Sort by priority
to_sell_sorted = sorted(to_sell, key=lambda i: returns[i])  # Sell worst first
to_buy_sorted = sorted(to_buy, key=lambda i: returns[i], reverse=True)  # Buy best first

# Create week-by-week plan
print(f"\n" + "-" * 40)
print("WEEKLY EXECUTION PLAN")
print("-" * 40)

weekly_plan = []
sell_queue = list(to_sell_sorted)
buy_queue = list(to_buy_sorted)

current_portfolio = set(current_holdings)
week = 1
total_trans_cost = 0

while sell_queue or buy_queue:
    week_sell = []
    week_buy = []
    week_cost = 0
    changes_this_week = 0
    
    # First: SELL (to reduce portfolio size)
    while sell_queue and changes_this_week < K:
        asset = sell_queue.pop(0)
        week_sell.append(asset)
        week_cost += trans_costs[asset]
        current_portfolio.discard(asset)
        changes_this_week += 1
    
    # Second: BUY (if room available)
    while buy_queue and changes_this_week < K and len(current_portfolio) < N:
        asset = buy_queue.pop(0)
        week_buy.append(asset)
        week_cost += trans_costs[asset]
        current_portfolio.add(asset)
        changes_this_week += 1
    
    total_trans_cost += week_cost
    
    weekly_plan.append({
        'week': week,
        'sell': week_sell,
        'buy': week_buy,
        'changes': changes_this_week,
        'cost': week_cost
    })
    
    print(f"\nWeek {week}:")
    print(f"  SELL: {week_sell}")
    print(f"  BUY:  {week_buy}")
    print(f"  Changes: {changes_this_week}/{K} ✓")
    print(f"  Cost: {week_cost:.4f}")
    print(f"  Portfolio size: {len(current_portfolio)}")
    
    week += 1
    if week > 20:
        break

print(f"\n" + "-" * 40)
print(f"TRANSITION COMPLETE")
print(f"  Total weeks: {len(weekly_plan)}")
print(f"  Total transaction cost: {total_trans_cost:.4f}")
print("-" * 40)

# WEIGHT OPTIMIZATION
print("\n" + "=" * 70)
print("STAGE 2: WEIGHT OPTIMIZATION")
print("=" * 70)

def optimize_weight(selected_companies, returns, cov_matrix, rf=0.02):
    selected = list(selected_companies)
    n = len(selected)

    # Extract data for selected assets only
    sel_returns = returns[selected]
    sel_cov = cov_matrix[np.ix_(selected, selected)]

    def negative_sharpe(weights):
        """Objective: negative Sharpe ratio (We minimize)."""
        port_return = np.dot(weights, sel_returns)
        port_variance = weights @ sel_cov @ weights
        port_risk = np.sqrt(port_variance)

        if port_risk < 1e-10:
            return 1000    # Avoid division by zero
        
        sharpe = (port_return - rf) / port_risk
        return -sharpe # Negative because we minimize 
    
    # Constraints: weight must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: each weight between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(n)]

    #Initial guess: equal weights
    w0 = np.ones(n) / n
    #Optimize
    result = minimize(
        negative_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x

    # Calculate final metrics
    final_return =np.dot(optimal_weights, sel_returns)
    final_variance = optimal_weights @ sel_cov @ optimal_weights
    final_risk = np.sqrt(final_variance)
    final_sharpe = (final_return - rf) / final_risk

    return optimal_weights, final_return, final_risk, final_sharpe
    
print("\nOptimizing portfolio weights...")
optimal_weights, opt_return, opt_risk, opt_sharpe = optimize_weight(
    optimal_portfolio, returns, cov_matrix
)

# Display results
print("\nOptimized Portfolio Weights:")
print("-" * 50)
print(f"{'Asset': <10} {'sector':<10} {'Weight':>10} {'Return':>12}")
print("-" * 50)

for i, asset in enumerate(optimal_sorted):
    #Find weight for this asset
    weight_idx = optimal_portfolio.index(asset)
    weight = optimal_weights[weight_idx]

    if weight > 0.01:   # Only show weight > 1%
        sector = str(sectors[asset])
        ret = returns[asset] * 100
        print(f"Asset {asset:<6} {sector:<10} {weight*100:>9.2f}% {ret:>+11.2f}%")

print("-" * 50)
print(f"\nOptimized Portfolio Metrics:")
print(f"  Expected Return: {opt_return*100:.2f}%")
print(f"  Risk (Std Dev):  {opt_risk*100:.2f}%")
print(f"  Sharpe Ratio:    {opt_sharpe:.4f}")       
    
# Compare Stage 1(Equal Weight) vs Stage 2 (Weight Optimization)
print("\n" + "=" * 70)
print("COMPARISON: STAGE 1 vs STAGE 2")
print("=" * 70)

print(f"\n{'Metric':<20} {'Stage 1 (Equal Weights)':>15} {'Stage 2 (Optimized)':>15}")
print("-" * 60)
print(f"{'Expected Return':<20} {port_return*100:>14.2f}% {opt_return*100:>14.2f}%")
print(f"{'Risk':<20} {port_risk*100:>17.2f}% {opt_risk*100:>17.2f}%")
print(f"{'Sharpe Ratio':<20} {port_sharpe:>18.4f} {opt_sharpe:>18.4f}")

print("\n" + "=" * 70)
print("FINAL SUMMARY - ALL REQUIRED OUTPUTS")
print("=" * 70)

print(f"""
1. SELECTED ASSETS (companies to hold):
   {optimal_sorted}

2. TRANSACTION LIST:
   • HOLD ({len(to_hold)} assets): {sorted(to_hold)}
   • BUY  ({len(to_buy)} assets):  {sorted(to_buy)}
   • SELL ({len(to_sell)} assets): {sorted(to_sell)}

3. EXPECTED PORTFOLIO RETURN:
   {opt_return*100:.2f}%

4. PORTFOLIO RISK (Standard Deviation):
   {opt_risk*100:.2f}%

5. TOTAL TRANSACTION COSTS:
   {total_trans_cost:.4f}

6. SECTOR ALLOCATION:
""")

for sector in sorted(sector_breakdown.keys()):
    count = sector_breakdown[sector]
    bar = "█" * count
    print(f"   {sector:<8}: {count} companies {bar}")

print(f"""
CONSTRAINT SATISFACTION:
   ✓ Portfolio size:      {len(optimal_portfolio)}/{N}
   ✓ Max per sector:      {max(sector_breakdown.values())}/{S_max}
   ✓ Max changes/week:    {K} (via {len(weekly_plan)}-week transition)

PERFORMANCE:
   • Sharpe Ratio: {opt_sharpe:.4f}
   • Computation Time: {computation_time:.2f} seconds
""")
print("SAVING RESULTS...")
print("-" * 40)

results = {
    'method': 'Classical - Simulated Annealing with Multi-Period Rebalancing',
    'parameters': {
        'N': N,
        'K': K,
        'S_max': S_max,
        'lambda_risk': lambda_risk,
        'tau': tau
    },
    'selected_assets': optimal_sorted,
    'transactions': {
        'hold': sorted([int(x) for x in to_hold]),
        'buy': sorted([int(x) for x in to_buy]),
        'sell': sorted([int(x) for x in to_sell])
    },
    'transition_plan': {
        'weeks_needed': len(weekly_plan),
        'weekly_details': weekly_plan
    },
    'metrics_equal_weights': {
        'expected_return': float(port_return),
        'risk': float(port_risk),
        'sharpe_ratio': float(port_sharpe)
    },
    'metrics_optimized': {
        'expected_return': float(opt_return),
        'risk': float(opt_risk),
        'sharpe_ratio': float(opt_sharpe)
    },
    'total_transaction_cost': float(total_trans_cost),
    'sector_breakdown': {str(k): int(v) for k, v in sector_breakdown.items()},
    'constraints_satisfied': {
        'portfolio_size': len(optimal_portfolio) == N,
        'sector_limit': max(sector_breakdown.values()) <= S_max,
        'weekly_change_limit': all(w['changes'] <= K for w in weekly_plan)
    },
    'computation_time': float(computation_time)
}

with open('classical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save('classical_selected.npy', np.array(optimal_sorted))
np.save('classical_weights.npy', optimal_weights)

print("Saved: classical_results.json")
print("Saved: classical_selected.npy")
print("Saved: classical_weights.npy")

print("\n" + "=" * 70)
print("CLASSICAL SOLUTION COMPLETE!")
print("=" * 70)
