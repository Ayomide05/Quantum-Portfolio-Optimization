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
def calculate_shrpe(selected, returns, cov_matrix, rf=0.02):
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

    def build_qubo_matrix(self, penalty_size=10.0, penalty_score=3.0):
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
                        Q[i, j] += penalty_score * 0.5
                        Q[j, i] += penalty_score * 0.5

        return Q, c
    def to_ising(self, penalty_size=10.0, penalty_sector=3.0):
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


    





