# Quantum Portfolio Optimization using Quantum Annealing 
# Unlike the Classical solution that uses thermal fluctuations to escape local minima 
# Quantum Annealing used quantum tunelling to escape local minima

import numpy as np
import json
import time
import os
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from neal import SimulatedAnnealingSampler
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False


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

#Problem Parameters
N = 15   #Target portfolio size (select 15 from 50)
K = 5    #Max position changes per week
S_max = 5   # Max assets per sector
risk_free_rate = 0.02
lambda_risk = 0.5    # Risk aversion parameter
tau = 0.1    # Transaction cost multiplier

current_holdings = set(i for i in range(n_assets) if prev_positions[i] == 1)

#Helper Functions
def calculate_sharpe(selected, returns, cov_matrix, rf=0.02):
    """Calculate Sharpe ratio for equal-weighted portfolio."""
    if len(selected) == 0:
        return -float('inf')
    
    selected = list(selected)
    n = len(selected)
    w = np.ones(n) / n

    port_return = np.dot(w, returns[selected])
    port_risk = np.sqrt(w @ cov_matrix[np.ix_(selected, selected)] @ w)

    if port_risk < 1e-10:
        return -float('inf')
    
    return (port_return - rf) / port_risk

def portfolio_metrics(selected, returns, cov_matrix):
    """Calculate portfolio return and risk with equal weights."""
    if len(selected) == 0:
        return 0, float('inf')
    selected = list(selected)
    n = len(selected)
    w = np.ones(n) / N

    port_return = np.dot(w, returns[selected])
    port_risk = np.sqrt(w @ cov_matrix[np.ix_(selected, selected)] @ w)

    return port_return, port_risk

def count_sectors(selected, sectors):
    """Count assets per sector."""
    counts = {}
    for i in selected:
        s = str(sectors[i])
        counts[s] = counts.get(s, 0) + 1
    return counts

def get_transactions(new_portfolio, prev_positions):
    """Calculate buy/sell/hold transactions."""
    new_set = set(new_portfolio)
    old_set = set(i for i in range(len(prev_positions)) if prev_positions[i] == 1)

    to_hold = new_set & old_set
    to_buy = new_set - old_set
    to_sell = old_set - new_set

    return to_hold, to_buy, to_sell

def calculate_transaction_cost(to_buy, to_sell, trans_costs):
    """Calculate total transaction cost."""
    total = sum(trans_costs[i] for i in to_buy) + sum(trans_costs[i] for i in to_sell)
    return total

return_scale = np.max(np.abs(returns))
cov_scale = np.max(np.abs(cov_matrix))
trans_scale = np.max(np.abs(trans_costs))

# STEP 1: QUBO FORMULATION
class PortfolioQUBO:
    """To build QUBO/Ising formulation for portfolio optimization, this class 
    will construct the mathematical model that maps our portfolio optimization
    problem to a form solvable by quantum annealing."""

    def __init__(self, returns, cov_matrix, prev_positions, trans_costs, sectors, N, S_max, lambda_risk, tau):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.prev_positions = prev_positions
        self.trans_costs = trans_costs
        self.sectors = sectors
        self.n_assets = len(returns)
        self.N = N
        self.S_max = S_max
        self.lambda_risk = lambda_risk
        self.tau = tau

        # Normalize for numerical stability
        self.return_scale = np.max(np.abs(returns)) if np.max(np.abs(returns)) > 0 else 1.0
        self.cov_scale = np.max(np.abs(cov_matrix)) if np.max(np.abs(cov_matrix)) > 0 else 1.0

    def build_qubo_matrix(self, penalty_size=1000.0, penalty_sector=5.0):
        n = self.n_assets
        Q = np.zeros((n, n))
        c = np.zeros(n)

        # Normalized coefficients
        norm_returns = self.returns / self.return_scale
        norm_cov = self.cov_matrix / self.cov_scale

     
        # Term 1: Return Maximization (our returns -Σ(μᵢ × xᵢ) has negative because we maximize returns by minimizing negatives)
        c -= norm_returns

        # TERM 2: RISK MINIMIZATION (λ × ΣΣ(σᵢⱼ × xᵢ × xⱼ))
        Q += self.lambda_risk * norm_cov

        # TERM 3: Transaction Costs
        for i in range(n):
            if self.prev_positions[i] == 1:
                # Currently held: reward keeping (penalize selling)
                c[i] -= self.tau * self.trans_costs[i] / self.return_scale
            else:
                # Not held: penalize buying
                c[i] += self.tau * self.trans_costs[i] / self.return_scale

        # Portfolio Size Constraint (P₁ × (Σxᵢ - N)² Expands to P₁ × [Σxᵢ² + 2ΣΣxᵢxⱼ - 2NΣxᵢ + N²] for binary variables: xᵢ² = xᵢ)
        for i in range(n):
            # Linear terms: xᵢ - 2Nxᵢ = (1-2N)xᵢ
            c[i] += penalty_size * (1 - 2 * self.N)

            # Quadractic terms: 2xᵢxⱼ for all pairs
            for j in range(i + 1, n):
                Q[i, j] += penalty_size * 2
                Q[j, i] += penalty_size * 2

        unique_sects = np.unique(self.sectors)
        for sect in unique_sects:
            sect_str = str(sect)
            sect_indices = [i for i in range(n) if str(self.sectors[i]) == sect_str]

            # Add penalty for pairs in the same sector which discourages over-concentration in any sector
            for i in sect_indices:
                for j in sect_indices:
                    if i < j:
                        Q[i, j] += penalty_sector * 0.5
                        Q[j, i] += penalty_sector * 0.5

        return Q, c
    
    def to_ising(self, penalty_size=1000.0, penalty_sector=5.0):
        Q, c = self.build_qubo_matrix(penalty_size, penalty_sector)
        n = self.n_assets
        h = np.zeros(n)     # Linear coefficients (local fields)
        J = np.zeros((n, n))   # Coupling coefficients
        offset = 0.0      # Constant energy offset

        # Convert each QUBO term to Ising
        for i in range(n):
            # Linear terms: cᵢxᵢ = cᵢ(sᵢ+1)/2 = (cᵢ/2)sᵢ + cᵢ/2
            h[i] += c[i] / 2.0
            offset += c[i] / 2.0

            # Diagonal quadratic: Qᵢᵢxᵢ² = Qᵢᵢxᵢ (binary)
            h[i] += Q[i, i] / 2.0
            offset += Q[i, i] / 2.0

            # Off-diagonal quadratic: Qᵢⱼxᵢxⱼ
            for j in range(i + 1, n):
                J[i, j] = Q[i, j] / 4.0
                h[i] += Q[i, j] / 4.0
                h[j] += Q[i, j] / 4.0
                offset += Q[i, j] / 4.0

        return h, J, offset


qubo = PortfolioQUBO(
    returns = returns,
    cov_matrix = cov_matrix,
    prev_positions = prev_positions,
    trans_costs = trans_costs,
    sectors = sectors,
    N = N,
    S_max = S_max,
    lambda_risk = lambda_risk,
    tau = tau
)

Q, c = qubo.build_qubo_matrix()
h, J, offset = qubo.to_ising()


print(f"QUBO Construction Complete:")
print(f"  Matrix Q shape:           {Q.shape}")
print(f"  Linear terms c:           {len(c)} coefficients")
print(f"  Ising local fields h:     {len(h)} coefficients")
print(f"  Ising couplings J:        {np.count_nonzero(J)} non-zero")


class QuantumAnnealingSolver:
    """ Quantum Annealing Solver for portfolio optimization. 
    This uses D-wave's SimulatedAnnealingSampler if available, otherwise 
    it falls back to custom quantum-inspired implementation"""

    def __init__(self, qubo, n_reads=1000, annealing_time=100):
        self.qubo = qubo
        self.n_reads = n_reads     # Number of annealing runs
        self.annealing_time =annealing_time   # Steps per run
        self.best_solution = None
        self.best_energy = float('inf')
        self.computation_time = 0
        self.all_energies = []

    def solve(self):
        # Run quantum annealing optimization
        n = self.qubo.n_assets
        Q, c = self.qubo.build_qubo_matrix()

        # Define energy function for validation
        def energy(x):
            # Calculate QUBO energy for solution x.
            return float(x @ Q @ x + c @ x)
        
        if DWAVE_AVAILABLE:
            print(f"  Backend:           D-Wave Neal (SimulatedAnnealingSampler)")
            return self._solve_dwave(Q, c, n)
        else:
           print(f"  Backend:           Custom Quantum-Inspired Annealer")
           return self._solve_custom(Q, c, n, energy)
        
    def _solve_dwave(self, Q, c, n):
        """Solving using D-Wave's Sampler."""
        # To build Binary Quandractic Model
        linear = {i: float(c[i] + Q[i, i]) for i in range(n)}
        quadractic = {}
        for i in range(n):
            for j in range(i + 1, n):     # Only look at j > i to avoid double counting
                val = float(Q[i, j] + Q[j, i])
                if abs(val) > 1e-10:
                    quadractic[(i, j)] = val
        
        bqm = dimod.BinaryQuadraticModel(linear, quadractic, 0.0, dimod.BINARY)
        sampler = SimulatedAnnealingSampler()

        print("\n Running D-Wave sampler...")
        start_time = time.time()

        response = sampler.sample(
            bqm,
            num_reads=self.n_reads,
            num_sweeps=self.annealing_time * 10
        )
        self.computation_time = time.time() - start_time

        # Collect All solutions and find the best one with exactly N assets
        all_solutions = []
        for record in response.record:
            sol = np.array([record.sample[i] for i in range(n)])
            n_selected = int(np.sum(sol))
            energy = record.energy
            all_solutions.append((sol, n_selected, energy))
            self.all_energies.append(energy)

        # Sort by: How close the solution is to N assets and the energy
        all_solutions.sort(key=lambda x: (abs(x[1] - self.qubo.N), x[2]))

        # Get best solution (Closeset to N assets, then lowest energy)
        best_sol, best_n, best_energy = all_solutions[0]

        self.best_solution = best_sol
        self.best_energy = best_energy

        # Collect all energies for analysis
        self.all_energies = [record.energy for record in response.record]
        
        return self.best_solution
    
    def _solve_custom(self, Q, c, n, energy_func):
        """This is a Custom quantum-inspired annealing implementation.
        It simulates the quantum annealing behaviour:
        - High "temperature" = strong quantum fluctuations (tunneling)
        - Multi-spin flips mimic quantum superposition
        - Exponential cooling mimics adiabatic evolution"""

        print("\n Running quantum-inspired annealer...")
        start_time = time.time()

        best_solution = None
        best_energy = float('inf')
        self.all_energies = []

        for read in range(self.n_reads):
            # Random Initial state (mimics initial superposition)
            np.random.seed(read)
            x = np.random.randint(0, 2, n).astype(float)
            current_energy = energy_func(x)
            
            run_best_x = x.copy()
            run_best_energy = current_energy
           
            # Annealing schedule
            T_initial = 100.0     # High "quantum flunctuations"
            T_final = 0.001       # Near-sero temperature
            n_steps = self.annealing_time * 100

            for step in range(n_steps):
                # Exponential cooling (adiabatic evolution)
                T = T_initial * (T_final / T_initial) ** (step / n_steps)

                # Quantum-inspired: allow multi-spin flips at high T which mimics quantum tunelling through barriers
                if T > 10:
                    n_flips = np.random.randint(1, 5)    #Up to 4 simultaneous flips
                elif T > 1:
                    n_flips = np.random.randint(1, 3)
                else:
                    n_flips = 1

                # Propose new state
                x_new = x.copy()
                flip_indices = np.random.choice(n, min(n_flips, n), replace=False)
                for idx in flip_indices:
                    x_new[idx]  = 1 - x_new[idx]

                new_energy = energy_func(x_new)
                delta_E = new_energy - current_energy

                # Metropolis acceptance (quantum tunneling probability) (This measures the difference in height)
                if delta_E < 0:    # if delta_E is Negative: The new spot is lower (downhill), accept it, we should always go downhill
                    accept = True
                else:  # This means it is positive and we can either accept or not depending on the value of T
                    #if T is High(Hot), we accept the "bad" move because it might lead to a better valley later and 
                    # if T is low we decline because we dont have energy to climb so we reject the move and stay where we are.
                    # Quantum tunneling: can escape local minima
                    accept = np.random.random() < np.exp(-delta_E / T)  # This creates the physics behaviour: When hot, jump anywhere (explore). When cold, refuse to go uphill (settle)
                    # np.random.random(): This is us rolling a dice to pick a random number between 0 and 1 that we will compare with np.exp(-delta_E / T) value, our chances of success
                if accept:
                    x = x_new
                    current_energy = new_energy

                    if current_energy < run_best_energy:
                        run_best_x = x.copy()
                        run_best_energy = current_energy

            self.all_energies.append(run_best_energy)

            if run_best_energy < best_energy:
                best_energy = run_best_energy
                best_solution = run_best_x.copy()

            # Progress update
            if (read + 1) % 200 == 0:
                print(f"    Completed {read+1}/{self.n_reads} reads | Best energy: {best_energy:.4f}")

        self.computation_time = time.time() - start_time
        self.best_solution = best_solution
        self.best_energy = best_energy

        return self.best_solution

# Run Quantum Annealing
solver = QuantumAnnealingSolver(
    qubo = qubo,
    n_reads=2000,
    annealing_time=200
)

solution = solver.solve()

# Extract selected assets from solution
raw_selected = [i for i in range(n_assets) if solution[i] > 0.5]

print(f"\n  Optimization Complete!")
print(f"  ──────────────────────────────────────────────────────────────")
print(f"  Computation time:  {solver.computation_time:.2f} seconds")
print(f"  Best energy:       {solver.best_energy:.4f}")
print(f"  Assets selected:   {len(raw_selected)} (raw solution)")
print(f"  Raw selection:     {sorted(raw_selected)}")

# Post-Processing (Enforcing Constraints)


def enforce_portfolio_constraints(raw_selected, returns, cov_matrix, sectors, prev_positions, N=15, S_max=5):
    """Adjust portfolio tob exactly N assets while respecting sector limits.
    Strategy:
    - If too many: remove assets with worst risk-adjusted returns
    - If too few: add assets with best risk-adjusted returns
    - Respect sector diversification constraint
    """            
    
    selected = list(raw_selected)
    

    def asset_score(idx):
        """Risk-adjusted return score."""
        vol = np.sqrt(cov_matrix[idx, idx])
        return returns[idx] / (vol + 1e-10)
    
    print(f"\n Raw selection: {len(selected)} assets")
    print(f"  Target:        {N} assets")

    #REMOVE excess assets
    removed = []
    while len(selected) > N:
        sector_counts = count_sectors(selected, sectors)

        # Find worst asset, preferring removal from over-represented sectors
        worst_idx = None
        worst_score = float('inf')

        for idx in selected:
            score = asset_score(idx)
            sect = str(sectors[idx])

            # Strongly prefer removing from over-represented sectors
            if sector_counts.get(sect, 0) > S_max:
                score -= 100

            if score < worst_score:
                worst_score = score
                worst_idx = idx

        selected.remove(worst_idx)
        removed.append(worst_idx)
    if removed:
       print(f"  Removed {len(removed)} assets: {removed}") 

    # ADD assets if needed
    added = []
    while len(selected) < N:
        sector_counts = count_sectors(selected, sectors)

        # Find best available asset respecting sector limits
        best_idx = None
        best_score = -float('inf')

        for idx in range(len(returns)):
            if idx in selected:
                continue

            sect = str(sectors[idx])

            # Skip if sector is at limit
            if sector_counts.get(sect, 0) >= S_max:
                continue

            score = asset_score(idx)

            # Bonus for currently held (lower transaction cost)
            if prev_positions[idx] == 1:
                score += 0.05 * abs(score)

            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            added.append(best_idx)
        else:
            # Relax sector constraint if no valid asset found
            for idx in range(len(returns)):
                if idx not in selected:
                    selected.append(idx)
                    added.append(idx)
                    print(f"    Warning: Added asset {idx} (sector constraint relaxed)")
                    break
    if added:
        print(f"    Added {len(added)} assets: {added}")

    return sorted(selected)

optimal_portfolio = enforce_portfolio_constraints(
    raw_selected, returns, cov_matrix, sectors, prev_positions, N=N, S_max=S_max
)

print(f"\n  Final portfolio: {optimal_portfolio}")
print(f"  Portfolio size:  {len(optimal_portfolio)} ✓")

sector_breakdown = count_sectors(optimal_portfolio, sectors)

# Weight Optimization
def optimize_weight(selected, returns, cov_matrix, rf=0.02):
    """We optimize portfolio weights using mean-variance optimization
    and we Maximize Sharpe ratio subject to weights summing to 1"""

    selected = list(selected)
    n = len(selected)

    sel_returns = returns[selected]
    sel_cov = cov_matrix[np.ix_(selected, selected)]

    def neg_sharpe(weights):
        port_return = np.dot(weights, sel_returns)
        port_variance = weights @ sel_cov @ weights
        port_risk = np.sqrt(port_variance)

        if port_risk < 1e-10:
            return 1000
        return -(port_return - rf) / port_risk
    
    # Constraints: Weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    # Bounds: Weights must sum to 1 (No shorting)
    bounds = [(0, 1) for _ in range(n)]

    w0 = np.ones(n) / n  # Start with equal weights

    #Optimize using SLSQP (Sequrntial Least Squares Programming)
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = result.x
    opt_return = np.dot(opt_weights, sel_returns)
    opt_risk = np.sqrt(opt_weights @ sel_cov @ opt_weights)
    opt_sharpe = (opt_return - rf) / opt_risk if opt_risk > 1e-10 else 0

    return opt_weights, opt_return, opt_risk, opt_sharpe

# Calculate metrics with equal weights (for comparison)
port_return_eq, port_risk_eq = portfolio_metrics(optimal_portfolio, returns, cov_matrix)
port_sharpe_eq = calculate_sharpe(optimal_portfolio, returns, cov_matrix)

# Optimize weights
optimal_weights, opt_return, opt_risk, opt_sharpe = optimize_weight(
    optimal_portfolio, returns, cov_matrix, risk_free_rate
)
  # Display optimized weights
print(f"\n  Optimized Asset Weights:")
print(f"  {'─'*55}")
print(f"  {'Asset':<8} {'Sector':<12} {'Weight':>10} {'Return':>12}")
print(f"  {'─'*55}")

for i, asset in enumerate(optimal_portfolio):
    w = optimal_weights[i]
    if w > 0.005:  # Show weights > 0.5%
        sect = str(sectors[asset])
        ret = returns[asset] * 100
        print(f"  Asset {asset:<3} {sect:<12} {w*100:>9.2f}% {ret:>+11.2f}%")

# Transaction Plan
to_hold, to_buy, to_sell = get_transactions(optimal_portfolio, prev_positions)
total_changes = len(to_buy) + len(to_sell)
total_trans_cost = calculate_transaction_cost(to_buy, to_sell, trans_costs) 

print(f"""
  Transaction Summary:
  Currently holding:  {len(current_holdings)} assets
  Target portfolio:   {len(optimal_portfolio)} assets
  
  HOLD ({len(to_hold):>2} assets):  {sorted(to_hold)}
  SELL ({len(to_sell):>2} assets):  {sorted(to_sell)}
  BUY  ({len(to_buy):>2} assets):  {sorted(to_buy)}
  
  Total changes:      {total_changes}
  Transaction cost:   {total_trans_cost:.4f}
""")
# Multi-week transition plan (respecting K constraint)
weeks_needed = int(np.ceil(total_changes / K))
print(f"  Weekly Execution Plan (max {K} changes per week):")
print(f"  Weeks needed: {weeks_needed}")
print(f"  {'─'*55}")       

# Sort by priority: sell worst fisrt, buy best first
to_sell_sorted = sorted(to_sell, key=lambda i: returns[i])
to_buy_sorted = sorted(to_buy, key=lambda i: returns[i], reverse=True)

weekly_plan = []
sell_queue = list(to_sell_sorted)
buy_queue = list(to_buy_sorted)
temp_portfolio = set(current_holdings)
week = 1

while sell_queue or buy_queue:
    week_actions = {'week': week, 'sell': [], 'buy': [], 'cost': 0.0}
    changes_this_week = 0

    # First: SELL
    while sell_queue and changes_this_week < K:
        asset = sell_queue.pop(0)
        week_actions['sell'].append(int(asset))
        week_actions['cost'] += float(trans_costs[asset])
        temp_portfolio.discard(asset)
        changes_this_week += 1

    # Then: BUY
    while buy_queue and changes_this_week < K and len(temp_portfolio) < N:
        asset = buy_queue.pop(0)
        week_actions['buy'].append(int(asset))
        week_actions['cost'] += float(trans_costs[asset])
        temp_portfolio.add(asset)
        changes_this_week += 1

    weekly_plan.append(week_actions)

    sell_str = str(week_actions['sell'])
    buy_str = str(week_actions['buy']) 

    print(f"  Week {week}: SELL {sell_str:<20} BUY {buy_str:<20} "
      f"Changes: {changes_this_week}/{K}  Cost: {week_actions['cost']:.4f}")
    
    week += 1
    if week > 20:
        break

# Final Perfromance Report
print("\n" + "═" * 60)
print("FINAL PERFORMANCE METRICS VS. EQUAL WEIGHTS")
print("═" * 60)
print(f"{'METRIC':<20} | {'EQUAL WEIGHTS':<15} | {'OPTIMIZED (MVO)':<15}")
print("-" * 60)

# Convert to percentages for display
print(f"{'Expected Return':<20} | {port_return_eq*100:>14.2f}% | {opt_return*100:>14.2f}%")
print(f"{'Annual Risk (Vol)':<20} | {port_risk_eq*100:>14.2f}% | {opt_risk*100:>14.2f}%")
print("-" * 60)
print(f"{'SHARPE RATIO':<20} | {port_sharpe_eq:>14.4f}  | {opt_sharpe:>14.4f}")
print("═" * 60)

# Save Result
results = {
'method': 'Quantum Annealing',
'algorithm': 'Quantum Annealing (Ising Hamiltonian)',
'description': 'Direct quantum analog of classical Simulated Annealing',
'parameters': {
    'N': N,
    'K': K,
    'S_max': S_max,
    'lambda_risk': lambda_risk,
    'tau': tau,
    'n_reads': solver.n_reads,
    'annealing_time': solver.annealing_time,
    'problem_size_qubits': n_assets
},
'selected_assets': optimal_portfolio,
'transactions': {
    'hold': sorted([int(x) for x in to_hold]),
    'buy': sorted([int(x) for x in to_buy]),
    'sell': sorted([int(x) for x in to_sell]),
    'total_changes': int(total_changes)
},
'transition_plan': {
    'weeks_needed': len(weekly_plan),
    'max_changes_per_week': K,
    'weekly_details': weekly_plan
},
'metrics_equal_weights': {
    'expected_return': float(port_return_eq),
    'risk': float(port_risk_eq),
    'sharpe_ratio': float(port_sharpe_eq)
},
'metrics_optimized': {
    'expected_return': float(opt_return),
    'risk': float(opt_risk),
    'sharpe_ratio': float(opt_sharpe)
},
'optimal_weights': {f'asset_{optimal_portfolio[i]}': float(optimal_weights[i]) 
                    for i in range(len(optimal_portfolio))},
'total_transaction_cost': float(total_trans_cost),
'sector_breakdown': {str(k): int(v) for k, v in sector_breakdown.items()},
'constraints_satisfied': {
    'portfolio_size': len(optimal_portfolio) == N,
    'sector_limit': max(sector_breakdown.values()) <= S_max,
    'weekly_change_limit': all(len(w['sell']) + len(w['buy']) <= K for w in weekly_plan)
},
'quantum_metrics': {
    'best_energy': float(solver.best_energy),
    'n_reads': solver.n_reads,
    'mean_energy': float(np.mean(solver.all_energies)) if solver.all_energies else None,
    'std_energy': float(np.std(solver.all_energies)) if solver.all_energies else None
},
'computation_time': float(solver.computation_time)
}

# Save to files
# Save to files
with open('quantum_results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save('quantum_selected.npy', np.array(optimal_portfolio))
np.save('quantum_weights.npy', optimal_weights)

print(f"\n  Saved: quantum_results.json")
print(f"  Saved: quantum_selected.npy")
print(f"  Saved: quantum_weights.npy")

    
    







