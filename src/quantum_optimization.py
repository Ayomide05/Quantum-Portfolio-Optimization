# Quantum Portfolio Optimization using Quantum Annealing 
# Unlike the Classical solution that uses thermal fluctuations to escape local minima 
# Quantum Annealing used quantum tunelling to escape local minima

import numpy as np
import json
import time
import os
from scipy.optimize import minimize
#from scipy.spatial.distance import mahalanobis
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

class HybridwarmStarter:
    """This generates diverse initial solutions using classical heuristics.
    So, instead of starting quantum annealing from random states, we use intelligenct classicla methods to generate goos starting point.
    which leads to faster convergence, Better exploration of solution space and Higher Quality final solutions
    Methods: - Greedy Sharpe: we will select top N by risk-adjusted return
    - Favor Holdings: Minimize transaction costs. 
    - Sector Balanced: Ensure diversification. 
    - Random variations: Explore different regions"""

    def __init__(self, returns, cov_matrix, sectors, prev_positions, N=15, S_max=5):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.sectors = sectors
        self.prev_positions = prev_positions
        self.n_assets = len(returns)
        self.N = N
        self.S_max = S_max
        self.warm_starts = []

    def check_sector_constraints(self, selected, new_asset):
        """Check if adding new_asset violates sector constraint."""
        sector_counts = count_sectors(selected, self.sectors)
        new_sector = str(self.sectors[new_asset])
        return sector_counts.get(new_sector, 0) < self.S_max
    
    def greedy_sharpe(self):
        """Method 1: Select top N assets by Sharpe ratio."""
        sharpe_scores = self.returns / (np.sqrt(np.diag(self.cov_matrix)) + 1e-10)
        sorted_indices = np.argsort(sharpe_scores)[::-1]

        selected = []
        for idx in sorted_indices:
            if len(selected) >= self.N:
                break
            if self.check_sector_constraints(selected, idx):
                selected.append(idx)
        return selected
    
    def favor_holdings(self):
        """Method 2: favor current holdings to minimize transaction costs."""
        current = [i for i in range(self.n_assets) if self.prev_positions[i] == 1]
        current_sorted = sorted(current, key=lambda i: self.returns[i], reverse=True)

        selected = []
        for idx in current_sorted:
            if len(selected) >= self.N:
                break
            if self.check_sector_constraints(selected, idx):
                selected.append(idx)
        # Fill remaining slots with best non-held assets
        if len(selected) < self.N:
            non_held = [i for i in range(self.n_assets) if self.prev_positions[i] == 0]
            non_held_sorted = sorted(non_held, key=lambda i: self.returns[i], reverse=True)

            for idx in non_held_sorted:
                if len(selected) >= self.N:
                    break
                if self.check_sector_constraints(selected, idx):
                    selected.append(idx)

        return selected
    def sector_balanced(self):
        """Method 3: select assets ensuring sector diversification."""
        unique_sects = np.unique(self.sectors)
        n_sectors = len(unique_sects)
        per_sector = max(1, self.N // n_sectors)

        selected = []

        for sect in unique_sects:
            sect_str = str(sect)
            sect_assets = [i for i in range(self.n_assets) if str(self.sectors[i]) == sect_str]
            sect_sorted = sorted(sect_assets, key=lambda i: self.returns[i], reverse=True)

            count = 0
            for idx in sect_sorted:
                if count >= per_sector or len(selected) >= self.N:
                    break
                selected.append(idx)
                count += 1
        # Fill remaining slots
        if len(selected) < self.N:
            remaining = [i for i in range(self.n_assets) if i not in selected]
            remaining_sorted = sorted(remaining, key=lambda i: self.returns[i], reverse=True)

            for idx in remaining_sorted:
                if len(selected) >= self.N:
                    break
                if self.check_sector_constraints(selected, idx):
                    selected.append(idx)

        return selected
    def random_variation(self, seed):
        """Method 4: Generate random variation for exploration."""
        np.random.seed(seed)
        noisy_returns = self.returns + np.random.randn(self.n_assets) * np.std(self.returns) * 0.2
        sorted_indices = np.argsort(noisy_returns)[::-1]

        selected = []
        for idx in sorted_indices:
            if len(selected) >= self.N:
                break
            if self.check_sector_constraints(selected, idx):
                selected.append(idx)
        return selected
    
    def generate_all(self, n_random=3):
        """Generate all warm start solutions."""
        self.warm_starts = []

        #Deterministic methods
        self.warm_starts.append(('Greedy Sharpe', self.greedy_sharpe()))
        self.warm_starts.append(('Favor Holdings', self.favor_holdings()))
        self.warm_starts.append(('Sector Balanced', self.sector_balanced()))  

        # Random variations for exploration
        for i in range(n_random):
            ws = self.random_variation(seed=42 + i)
            self.warm_starts.append((f'Random Variation {i+1}', ws))
        return self.warm_starts  

    def to_binary_vector(self, selected):
        """Convert selected to binary vector."""
        x = np.zeros(self.n_assets)
        for i in selected:
            x[i] = 1
        return x

    def get_report(self):
        """Get warm start analysis report."""
        report = []
        for name, selection in self.warm_starts:
            sharpe = calculate_sharpe(selection, self.returns, self.cov_matrix)

            # Calculate transaction cost for this selection
            to_hold, to_buy, to_sell = get_transactions(selection, self.prev_positions)
            trans_cost = sum(trans_costs[i] for i in to_buy) + sum(trans_costs[i] for i in to_sell)

            report.append({
                'method': name,
                'selection': sorted(selection),
                'sharpe': sharpe,
                'n_assets': len(selection),
                'transaction_cost': trans_costs,
                'changes': len(to_buy) + len(to_sell)
            }) 
        return report
# Generate warm starts
warm_starter = HybridwarmStarter(
    returns=returns,
    cov_matrix=cov_matrix,
    sectors=sectors,
    prev_positions=prev_positions,
    N=N,
    S_max=S_max
)             
warm_starts = warm_starter.generate_all(n_random=3)
warm_start_report = warm_starter.get_report()    

class RebalancingTrigger:
    """This monitors portfolio drift and determines when rebalancing is needed. 
    It used Mahalanobis distance to measure drift, which accounts for correlations between assets (unlike simple Euclidean distance).
    Benefits: - Avoid unnecessary rebalancing (save transaction costs)
    - Don't let portfolio drift too far (manage risk)
    - Smarter than fixed schedules (monthly, quarterly, etc.)"""

    def __init__(self, optimal_weights, cov_matrix, threshold=0.5):
        self.optimal_weights = optimal_weights
        self.cov_matrix = cov_matrix
        self.threshold = threshold
        self.history = []
    
    def calculate_drift_euclidean(self, current_weights):
        """Calculate simple Euclidean distance (baseline)."""
        diff = current_weights - self.optimal_weights
        return np.sqrt(np.sum(diff ** 2))
    
    def calculate_drift_mahalanobis(self, current_weights):
        """
        Calculate Mahalanobis distance (risk-adjusted drift).
        
        Mahalanobis distance accounts for correlations:
        - Drift in correlated assets is less concerning (they move together)
        - Drift in uncorrelated assets is more concerning (diversification loss)
        
        Formula: d = √[(w - w*)ᵀ × Σ⁻¹ × (w - w*)]
        """
        diff = current_weights - self.optimal_weights
        n = len(current_weights)
        
        # Regularize covariance for numerical stability
        cov_subset = self.cov_matrix[:n, :n]
        cov_reg = cov_subset + np.eye(n) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov_reg)
            drift = np.sqrt(np.abs(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            # Fallback to Euclidean if inversion fails
            drift = self.calculate_drift_euclidean(current_weights)
        
        return drift
    def check_trigger(self, current_weights):
        """
        Check if rebalancing should be triggered.
        
        Returns:
            should_rebalance (bool): Whether to rebalance
            drift (float): Current drift value
            urgency (str): Urgency level description
        """
        drift = self.calculate_drift_mahalanobis(current_weights)
        
        if drift > self.threshold * 2:
            return True, drift, "CRITICAL - Rebalance immediately!"
        elif drift > self.threshold:
            return True, drift, "WARNING - Rebalance recommended"
        elif drift > self.threshold * 0.5:
            return False, drift,  "MONITOR - Drift increasing"
        else:
            return False, drift, "STABLE - Portfolio on track"

    def simulate_drift(self, days=30, daily_vol=0.02, seed=42):
        """
        Simulate portfolio drift over time. This demonstrates how the trigger system would work in practice.
        Args:
            days: Number of days to simulate
            daily_vol: Daily return volatility
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        current = self.optimal_weights.copy()
        self.history = []
        
        for day in range(days):
            # Simulate random daily returns
            daily_returns = np.random.randn(len(current)) * daily_vol
            
            # Update weights (prices change, weights drift)
            current = current * (1 + daily_returns)
            current = current / current.sum()  # Renormalize to sum to 1
            
            # Check trigger
            should_rebal, drift, urgency = self.check_trigger(current)
            
            self.history.append({
                'day': day + 1,
                'drift': drift,
                'urgency': urgency,
                'should_rebalance': should_rebal,
                'weights': current.copy()
            })
            
            # If rebalancing triggered, reset to optimal
            if should_rebal:
                current = self.optimal_weights.copy()
        
        return self.history 
    def get_report(self):
        """Get rebalancing analysis report."""
        if not self.history:
            return None
        
        drifts = [h['drift'] for h in self.history]
        trigger_days = [h['day'] for h in self.history if h['should_rebalance']]
        
        return {
            'simulation_days': len(self.history),
            'max_drift': float(max(drifts)),
            'avg_drift': float(np.mean(drifts)),
            'min_drift': float(min(drifts)),
            'trigger_days': trigger_days,
            'n_triggers': len(trigger_days),
            'threshold': self.threshold,
            'rebalancing_frequency': f"Every {len(self.history) / max(1, len(trigger_days)):.1f} days on average" if trigger_days else "No rebalancing needed"
        }    
    
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


class WarmStartQuantumAnnealer:
    """ Quantum Annealing Solver for portfolio optimization with hybrid warm starting. 
    This uses D-wave's SimulatedAnnealingSampler if available, otherwise 
    it falls back to custom quantum-inspired implementation
    We use classical heuristics to generate good initial solutions, then runs quantum annealing from each starting point."""

    def __init__(self, qubo, warm_starts, n_reads_per_start=350, annealing_time=100):
        self.qubo = qubo
        self.warm_starts = warm_starts
        self.n_reads_per_start = n_reads_per_start     # Number of annealing runs
        self.annealing_time =annealing_time   # Steps per run
        self.best_solution = None
        self.best_energy = float('inf')
        self.best_method = None
        self.computation_time = 0
        self.all_results = []

    def solve(self):
        # Run quantum annealing optimization
        n = self.qubo.n_assets
        Q, c = self.qubo.build_qubo_matrix()

        # Define energy function for validation
        def energy(x):
            # Calculate QUBO energy for solution x.
            return float(x @ Q @ x + c @ x)
        total_reads = len(self.warm_starts) * self.n_reads_per_start

        start_time = time.time()
        
        if DWAVE_AVAILABLE:
            self._solve_dwave(Q, c, n, energy)
        else:
           self._solve_custom(Q, c, n, energy)

        self.computation_time = time.time() - start_time

        return self.best_solution
        
    def _solve_dwave(self, Q, c, n, energy_func):
        """Solving using D-Wave with warm starts."""
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

        print("\n Running D-wave quantum annealing with warm starts...")
        for ws_idx, (ws_name, ws_selection) in enumerate(self.warm_starts):
            print(f"  [{ws_idx + 1}/{len(self.warm_starts)}] {ws_name}...", end=" ")

            # Create initial state from warm start
            initial_state = {i: (1 if i in ws_selection else 0) for i in range(n)}
            
            response = sampler.sample(
                bqm,
                num_reads=self.n_reads_per_start,
                num_sweeps=self.annealing_time * 10,
                initial_states=initial_state
            )
            # Process results
            ws_best_energy = float('inf')
            ws_best_solution = None

            for record in response.record:
                sol = np.array([record.sample[i] for i in range(n)])
                energy = record.energy
                n_selected = int(np.sum(sol))
                
                # Prefer solutions with exactly N assets
                adjusted_energy = energy + 1000 * abs(n_selected - self.qubo.N)

                if adjusted_energy < ws_best_energy:
                    ws_best_energy = adjusted_energy
                    ws_best_solution = sol

            ws_n_selected = int(np.sum(ws_best_solution))
            print(f"Best: {ws_n_selected} assets, Energy: {ws_best_energy:.2f}")

            self.all_results.append({
            'method': ws_name,
            'solution': ws_best_solution,
            'energy': ws_best_energy,
            'n_selected': ws_n_selected
            })

            if ws_best_energy < self.best_energy:
                self.best_energy = ws_best_energy
                self.best_solution = ws_best_solution
                self.best_method = ws_name

        
    
    def _solve_custom(self, Q, c, n, energy_func):
        """This is a Custom quantum-inspired annealing implementation with warm starts.
        It simulates the quantum annealing behaviour:
        - High "temperature" = strong quantum fluctuations (tunneling)
        - Multi-spin flips mimic quantum superposition
        - Exponential cooling mimics adiabatic evolution"""

        print("\n Running quantum-inspired annealer with warm starts...")
        for ws_idx, (ws_name, ws_selection) in enumerate(self.warm_starts):
            print(f"  [{ws_idx + 1}/{len(self.warm_starts)}] {ws_name}...", end=" ")
        
        ws_best_energy = float('inf')
        ws_best_solution = None

        for read in range(self.n_reads_per_start):
            # Initialize fromwarm start
            x = np.zeros(n)
            for i in ws_selection:
                x[i] = 1
            current_energy = energy_func(x)
            best_x = x.copy()
            best_energy = current_energy
           
            # Annealing schedule
            T_initial = 50.0     # High "quantum flunctuations"
            T_final = 0.001       # Near-sero temperature
            n_steps = self.annealing_time * 50

            for step in range(n_steps):
                # Exponential cooling (adiabatic evolution)
                T = T_initial * (T_final / T_initial) ** (step / n_steps)

                # Swap move (maintains count near N)
                ones = np.where(x == 1)[0]
                zeros = np.where(x == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                        x_new = x.copy()
                        idx_out = np.random.choice(ones)
                        idx_in = np.random.choice(zeros)
                        x_new[idx_out] = 0
                        x_new[idx_in] = 1
                        
                        new_energy = energy_func(x_new)
                        delta = new_energy - current_energy
                        
                        if delta < 0 or np.random.random() < np.exp(-delta / T):
                            x = x_new
                            current_energy = new_energy
                            
                            if current_energy < best_energy:
                                best_x = x.copy()
                                best_energy = current_energy
                
                n_selected = int(np.sum(best_x))
                adjusted_energy = best_energy + 10000 * abs(n_selected - self.qubo.N)

                if adjusted_energy < ws_best_energy:
                    ws_best_energy = adjusted_energy
                    ws_best_solution = best_x
            
            ws_n_selected = int(np.sum(ws_best_solution))
            print(f"Best: {ws_n_selected} assets, Energy: {ws_best_energy:.2f}")
            
            self.all_results.append({
                'method': ws_name,
                'solution': ws_best_solution,
                'energy': ws_best_energy,
                'n_selected': ws_n_selected
            })
            
            if ws_best_energy < self.best_energy:
                self.best_energy = ws_best_energy
                self.best_solution = ws_best_solution
                self.best_method = ws_name
        
                
# Run quantum annealing with warm starts
warm_annealer = WarmStartQuantumAnnealer(
    qubo=qubo,
    warm_starts=warm_starts,
    n_reads_per_start=350,
    annealing_time=100
)
solution = warm_annealer.solve()
raw_selected = [i for i in range(n_assets) if solution[i] > 0.5]

print(f"""Quantum Annealing Complete!
  Computation time:   {warm_annealer.computation_time:.2f} seconds
  Best energy:        {warm_annealer.best_energy:.4f}
  Best warm start:    {warm_annealer.best_method}
  Assets selected:    {len(raw_selected)} (raw solution)
""")

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
        score = returns[idx] / (vol + 1e-10)
        if prev_positions[idx] == 1:
            score += 0.05 * abs(score)  # Bonus for current holdings
        return score
    
    print(f"\n Raw selection: {len(selected)} assets")
    print(f"  Target:        {N} assets")

    #REMOVE excess assets
    removed = []
    while len(selected) > N:
        sector_counts = count_sectors(selected, sectors)
        # Remove worst, preferring over-represented sectors
        worst = min(selected, key=lambda idx: 
                asset_score(idx) - 1000 * (sector_counts.get(str(sectors[idx]), 0) > S_max))
        selected.remove(worst)
        removed.append(worst)

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
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            added.append(best_idx)
        else:
            break
          
    if added:
        print(f"    Added {len(added)} assets: {added}")
    if not removed and not added:
        print(f"  ✓ Solution already meets constraints!")

    return sorted(selected)

optimal_portfolio = enforce_portfolio_constraints(
    raw_selected, returns, cov_matrix, sectors, prev_positions, N=N, S_max=S_max
)

print(f"\n  Final portfolio: {optimal_portfolio}")
print(f"  Portfolio size:  {len(optimal_portfolio)} ✓")

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

# Rebalancing Trigger Simulation
# Get covariance for selected assets
selected_cov = cov_matrix[np.ix_(optimal_portfolio, optimal_portfolio)]

rebal_trigger = RebalancingTrigger(
    optimal_weights=optimal_weights,
    cov_matrix=selected_cov,
    threshold=0.5
)
# Simulate 30 days
drift_history = rebal_trigger.simulate_drift(days=30, daily_vol=0.02)
rebal_report = rebal_trigger.get_report()

for day_data in drift_history[::3]:  # Show every 3rd day
    print(f"   {day_data['day']:2d} │ {day_data['drift']:.4f} │ {day_data['urgency']}")

if rebal_report['trigger_days']:
    print(f"\n  ⚠ Rebalancing triggered on days: {rebal_report['trigger_days']}")
else:
    print(f"\n  ✅ No rebalancing needed during simulation period")

# Transaction Plan
to_hold, to_buy, to_sell = get_transactions(optimal_portfolio, prev_positions)
total_changes = len(to_buy) + len(to_sell)
total_trans_cost = calculate_transaction_cost(to_buy, to_sell, trans_costs) 

sector_breakdown = count_sectors(optimal_portfolio, sectors)

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

# Save files
output_dir = os.path.dirname(os.path.abspath(__file__))

# Save numpy arrays (these always work)
np.save(os.path.join(output_dir, 'quantum_selected.npy'), np.array(optimal_portfolio))
np.save(os.path.join(output_dir, 'quantum_weights.npy'), optimal_weights)

# Save results as simple text file instead of JSON
with open(os.path.join(output_dir, 'quantum_results.txt'), 'w') as f:
    f.write("QUANTUM PORTFOLIO OPTIMIZATION RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Selected Assets: {optimal_portfolio}\n")
    f.write(f"Portfolio Size: {len(optimal_portfolio)}\n\n")
    f.write(f"Expected Return: {float(opt_return)*100:.2f}%\n")
    f.write(f"Risk (Volatility): {float(opt_risk)*100:.2f}%\n")
    f.write(f"Sharpe Ratio: {float(opt_sharpe):.4f}\n\n")
    f.write(f"Transaction Cost: {float(total_trans_cost):.4f}\n")
    f.write(f"Weeks Needed: {len(weekly_plan)}\n\n")
    f.write("Innovations:\n")
    f.write(f"  - Hybrid Warm Start: {len(warm_starts)} methods, Best: {warm_annealer.best_method}\n")
    f.write(f"  - Rebalancing Trigger: {rebal_report['n_triggers']} triggers in 30-day simulation\n")

print(f"""
  Saved Files:
    • quantum_selected.npy   - Selected asset indices
    • quantum_weights.npy    - Optimal weights  
    • quantum_results.txt    - Summary results
""")