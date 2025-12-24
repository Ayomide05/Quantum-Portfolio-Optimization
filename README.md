# Quantum Portfolio Optimization

## Africa Quantum Computing Hackathon

A comprehensive portfolio optimization solution using both **Classical** (Simulated Annealing) and **Quantum** (QAOA) approaches to solve a real-world financial optimization problem.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Mathematical Background](#mathematical-background)
3. [Data Preparation](#data-preparation)
   - [Covariance Matrix Estimation](#covariance-matrix-estimation)
4. [Solution Approach](#solution-approach)
5. [Classical Solution](#classical-solution)
6. [Quantum Solution](#quantum-solution)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [File Structure](#file-structure)
10. [References](#references)

---

## Problem Statement

### The Challenge

You are a quantum finance analyst at Africa Quantum Consortium. The company manages a portfolio of **50 assets** and needs to rebalance it weekly. Due to transaction costs and market impact, you can only change a limited number of positions each week.

### Objective

Find the optimal set of assets to buy/sell/hold that **maximizes expected returns** while:

1. Staying within risk tolerance limits
2. Minimizing transaction costs
3. Maintaining sector diversification constraints
4. Limiting position changes to no more than K assets per period

### Constraints

| Constraint | Description | Value |
|------------|-------------|-------|
| Portfolio Size (N) | Number of assets to hold | 15 |
| Sector Limit (S_max) | Maximum assets per sector (35%) | 5 |
| Change Limit (K) | Maximum position changes per week | 5 |
| Transaction Costs | Fees for buying/selling | Minimize |

---

## Mathematical Background

### 1. Portfolio Return

The expected return of a portfolio is the weighted average of individual asset returns:

```
Portfolio Return = Σ(wi × μi)

Where:
- wi = weight of asset i in portfolio
- μi = expected return of asset i
```

For equal-weighted portfolios (our Stage 1):
```
wi = 1/N for all assets
Portfolio Return = (1/N) × Σμi = mean(returns of selected assets)
```

### 2. Portfolio Risk (Variance and Covariance)

#### What is Variance?

Variance measures how much an asset's returns deviate from its average:

```
Variance(σ²) = (1/n) × Σ(ri - μ)²

Where:
- ri = return at time i
- μ = average return
- n = number of observations
```

**High variance = High risk** (returns are unpredictable)
**Low variance = Low risk** (returns are stable)

#### What is Covariance?

Covariance measures how two assets move together:

```
Cov(A, B) = (1/n) × Σ(rA,i - μA)(rB,i - μB)

Where:
- rA,i = return of asset A at time i
- μA = average return of asset A
```

**Positive covariance**: Assets move in the same direction
**Negative covariance**: Assets move in opposite directions
**Zero covariance**: Assets move independently

#### Why Covariance Matters for Portfolios

If you hold two assets that always move together (high positive covariance), you're not really diversified. When one goes down, the other goes down too.

If you hold assets with negative covariance, when one goes down, the other might go up, reducing your overall risk.

#### The Covariance Matrix

For N assets, we need to know the covariance between every pair. This creates an N×N matrix:

```
         Asset 0   Asset 1   Asset 2   ...
Asset 0  [Var(0)   Cov(0,1)  Cov(0,2)  ...]
Asset 1  [Cov(1,0) Var(1)    Cov(1,2)  ...]
Asset 2  [Cov(2,0) Cov(2,1)  Var(2)    ...]
...      [...      ...       ...       ...]
```

**Properties:**
- Diagonal elements = Variance of each asset
- Off-diagonal elements = Covariance between pairs
- Symmetric: Cov(A,B) = Cov(B,A)

#### How We Calculate the Covariance Matrix

```python
import numpy as np

# Returns data: each row is a time period, each column is an asset
# Shape: (T time periods, N assets)
returns_data = np.array([...])

# NumPy calculates it for us:
# rowvar=False means columns are variables (assets)
cov_matrix = np.cov(returns_data, rowvar=False)
```

#### Portfolio Variance (Risk)

For a portfolio with weights w, the total risk is:

```
Portfolio Variance = wᵀ × Σ × w

Where:
- w = vector of weights [w1, w2, ..., wN]
- Σ = covariance matrix
- wᵀ = transpose of w
```

In Python:
```python
portfolio_variance = weights @ cov_matrix @ weights
portfolio_risk = np.sqrt(portfolio_variance)  # Standard deviation
```

### 3. Sharpe Ratio

The Sharpe ratio measures **return per unit of risk**:

```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Risk

               (μp - rf)
            = ───────────
                  σp
```

**Higher Sharpe = Better** (more return for each unit of risk taken)

| Sharpe Ratio | Interpretation |
|--------------|----------------|
| < 1.0 | Poor |
| 1.0 - 2.0 | Acceptable |
| 2.0 - 3.0 | Good |
| > 3.0 | Excellent |

### 4. The Optimization Problem

We want to find which assets to include (binary decision) that maximizes Sharpe ratio while satisfying constraints.

#### QUBO Formulation

The problem can be written as a Quadratic Unconstrained Binary Optimization (QUBO):

```
Minimize: -Σ(μi × xi) + λ × Σ(σij × xi × xj) + τ × Σ|xi - xi_prev|

Subject to:
- Σxi = N                    (portfolio size)
- Σ(sector_k × xi) ≤ S_max   (sector limits)
- Σ|xi - xi_prev| ≤ K        (change limits)

Where:
- xi ∈ {0, 1}    (1 = hold, 0 = don't hold)
- μi            = expected return of asset i
- σij           = covariance between assets i and j
- λ             = risk aversion parameter
- τ             = transaction cost multiplier
- xi_prev       = previous position (0 or 1)
```

#### Ising Model Mapping

For quantum computing, we map to an Ising Hamiltonian:

```
H = Σ(hi × si) + Σ(Jij × si × sj)

Where:
- si ∈ {-1, +1}
- Relationship: si = 2xi - 1
```

---

## Data Preparation

### Input Data

| File | Description | Shape |
|------|-------------|-------|
| `expected_returns.npy` | Expected return for each asset | (50,) |
| `covariance_matrix.npy` | Covariance between all asset pairs | (50, 50) |
| `previous_positions.npy` | Current holdings (0 or 1) | (50,) |
| `transaction_costs.npy` | Cost to buy/sell each asset | (50,) |
| `sector_list.npy` | Sector label for each asset | (50,) |
| `volatilities.npy` | Volatility of each asset | (50,) |

### Data Statistics

```
Total assets: 50
Currently holding: 36 assets
Sectors: ['CONS', 'ENERGY', 'FIN', 'HEALTH', 'TECH']
```

### Covariance Matrix Estimation

The original dataset did not include historical price data or a pre-computed covariance matrix. We derived the covariance matrix using market capitalization data and sector information through a three-step process.

#### Step 1: Volatility Estimation from Market Cap

There is a well-documented empirical relationship in finance: **smaller companies tend to have higher volatility** than larger companies. This is because:
- Smaller companies are more sensitive to market shocks
- They often have less diversified revenue streams
- Their stocks typically have lower liquidity

We used this relationship to estimate volatility from market cap:

```python
# Normalize market caps using log transformation
log_caps = np.log(market_caps + 1)
normalized_caps = (log_caps - log_caps.min()) / (log_caps.max() - log_caps.min())

# Map to volatility range (inverse relationship)
base_volatility = 0.20    # 20% for largest companies
volatility_range = 0.40   # Additional 40% for smallest companies

volatilities = base_volatility + volatility_range * (1 - normalized_caps)
```

**Result:**
- Largest market cap → ~20% annual volatility (lower risk)
- Smallest market cap → ~60% annual volatility (higher risk)

| Market Cap Rank | Volatility (σ) |
|-----------------|----------------|
| Top 10 | 20-25% |
| Middle | 35-45% |
| Bottom 10 | 50-60% |

#### Step 2: Correlation Matrix Construction

We built the correlation matrix based on sector membership, using the well-known principle that **assets in the same sector tend to move together**:

```python
correlation_matrix = np.zeros((n_assets, n_assets))

for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            correlation_matrix[i, j] = 1.0              # Self-correlation
        elif sectors[i] == sectors[j]:
            correlation_matrix[i, j] = 0.60             # Same sector
        else:
            correlation_matrix[i, j] = 0.20             # Different sectors
```

**Correlation Values:**

| Relationship | Correlation (ρ) | Rationale |
|--------------|-----------------|-----------|
| Same asset | 1.00 | Perfect correlation with itself |
| Same sector | 0.60 | Assets in same industry move together |
| Different sector | 0.20 | General market correlation |

**Sector Distribution:**

| Sector | Count | Assets |
|--------|-------|--------|
| ENERGY | 13 | Assets 3, 6, 7, 9, 10, ... |
| CONS | 10 | Assets 0, 17, 22, 30, ... |
| HEALTH | 10 | Assets 4, 5, 8, 24, ... |
| FIN | 10 | Assets 1, 25, 33, 39, ... |
| TECH | 7 | Assets 2, 18, 38, 41, ... |

#### Step 3: Covariance Matrix Calculation

The covariance matrix is computed from volatilities and correlations:

```
Covariance(i, j) = ρᵢⱼ × σᵢ × σⱼ

Where:
- ρᵢⱼ = correlation between assets i and j
- σᵢ  = volatility (standard deviation) of asset i
- σⱼ  = volatility (standard deviation) of asset j
```

**Special case (diagonal elements = variance):**
```
Variance(i) = σᵢ × σᵢ = σᵢ²
```

**In Python:**
```python
covariance_matrix = np.zeros((n_assets, n_assets))

for i in range(n_assets):
    for j in range(n_assets):
        covariance_matrix[i, j] = (
            correlation_matrix[i, j] * 
            volatilities[i] * 
            volatilities[j]
        )
```

#### Validation

The resulting covariance matrix was validated for correctness:

| Property | Expected | Achieved |
|----------|----------|----------|
| Symmetric | Σᵢⱼ = Σⱼᵢ | ✅ |
| Positive Semi-Definite | All eigenvalues ≥ 0 | ✅ |
| Diagonal = Variance | Σᵢᵢ = σᵢ² | ✅ |
| Shape | (50, 50) | ✅ |

**Matrix Statistics:**
```
Variance range:  0.0400 (4%) to 0.3600 (36%)
Covariance range: 0.0080 to 0.2160
All eigenvalues: positive (matrix is valid)
```

#### Why This Approach?

This estimation method is reasonable because:

1. **Market cap → Volatility** is a well-established empirical relationship (Fama-French research)
2. **Sector-based correlation** captures real market dynamics where industry peers move together
3. **The matrix is mathematically valid** (positive semi-definite) for optimization
4. **Values are realistic** - volatilities and correlations fall within typical market ranges

In production, you would use historical returns data to compute the actual covariance matrix. Our estimation provides a reasonable approximation for the hackathon dataset.

---

## Solution Approach

### Why Two Stages?

#### Stage 1: Asset Selection (Binary Problem)

**Question:** WHICH 15 assets should we hold?

**Challenge:** There are C(50,15) = 2.25 trillion possible combinations!

**Solution:** Use optimization algorithms (Simulated Annealing or QAOA) to find a good solution without checking all possibilities.

#### Stage 2: Weight Optimization (Continuous Problem)

**Question:** HOW MUCH should we invest in each selected asset?

**Method:** Once we have 15 assets, use convex optimization (Scipy SLSQP) to find optimal weights that maximize Sharpe ratio.

### Multi-Period Rebalancing

**The Problem:**
- Currently holding: 36 assets
- Target: 15 assets
- Need to change: 31 positions (26 sells + 5 buys)
- Constraint: Maximum 5 changes per week

**The Solution:** Create a transition plan over multiple weeks!

```
Week 1: Sell 5 worst assets     → Portfolio: 31 assets
Week 2: Sell 5 more            → Portfolio: 26 assets
Week 3: Sell 5 more            → Portfolio: 21 assets
Week 4: Sell 5 more            → Portfolio: 16 assets
Week 5: Sell 5 more            → Portfolio: 11 assets
Week 6: Sell 1, Buy 4          → Portfolio: 14 assets
Week 7: Buy 1                  → Portfolio: 15 assets ✓
```

Each week respects K=5 constraint!

---

## Classical Solution

### Algorithm: Simulated Annealing

Simulated Annealing is inspired by the physical process of annealing in metallurgy, where metals are heated and slowly cooled to reduce defects.

#### How It Works

```
1. Start with a random valid portfolio
2. Make a small random change (swap one asset)
3. If the change is better → Accept it
4. If the change is worse → Accept with probability e^(-ΔE/T)
5. Gradually reduce temperature T
6. As T decreases, fewer bad moves are accepted
7. Eventually converge to a good solution
```

#### Why It Works

- **High temperature (early):** Accepts many changes, explores widely, escapes local minima
- **Low temperature (later):** Only accepts improvements, refines the solution

#### Pseudocode

```python
def simulated_annealing():
    current = random_valid_portfolio()
    best = current
    temperature = 100.0
    
    for iteration in range(max_iterations):
        # Make small change
        neighbor = swap_one_asset(current)
        
        # Calculate improvement
        delta = sharpe(neighbor) - sharpe(current)
        
        if delta > 0:
            # Better - always accept
            current = neighbor
        else:
            # Worse - sometimes accept
            probability = exp(delta / temperature)
            if random() < probability:
                current = neighbor
        
        # Update best
        if sharpe(current) > sharpe(best):
            best = current
        
        # Cool down
        temperature *= 0.9995
    
    return best
```

#### Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Temperature | 100.0 | Starting randomness |
| Cooling Rate | 0.9995 | How fast to cool |
| Iterations | 20,000 | Number of steps |
| Final Temperature | 0.01 | Minimum temperature |

---

## Quantum Solution

### Algorithm: QAOA (Quantum Approximate Optimization Algorithm)

*[To be implemented]*

QAOA is a hybrid quantum-classical algorithm designed for combinatorial optimization problems.

#### How It Works

```
1. Encode the problem as a cost Hamiltonian H_C
2. Initialize qubits in superposition |+⟩^n
3. Apply Cost Unitary: e^(-iγH_C)
4. Apply Mixer Unitary: e^(-iβH_M)
5. Repeat steps 3-4 for p layers
6. Measure to get a solution
7. Classically optimize γ and β parameters
8. Repeat until convergence
```

#### Why Quantum?

For combinatorial optimization:
- Classical: Must explore solutions sequentially
- Quantum: Explores all solutions simultaneously via superposition
- Potential speedup for large problem sizes

---

## Results

### Classical Solution Results

| Metric | Equal Weights | Optimized Weights |
|--------|---------------|-------------------|
| Expected Return | 69.69% | 101.96% |
| Risk (Std Dev) | 23.90% | 20.64% |
| Sharpe Ratio | 2.8323 | **4.8435** |

### Selected Portfolio

```
Assets: [2, 13, 16, 17, 18, 19, 21, 32, 34, 36, 37, 39, 42, 43, 47]
```

### Optimal Weights (Top Holdings)

| Asset | Sector | Weight | Expected Return |
|-------|--------|--------|-----------------|
| 19 | ENERGY | 35.24% | +152.80% |
| 17 | CONS | 32.96% | +79.43% |
| 39 | FIN | 15.64% | +78.24% |
| 18 | TECH | 15.45% | +59.28% |

### Transaction Plan

| Week | SELL | BUY | Changes | Cost |
|------|------|-----|---------|------|
| 1 | [44, 30, 24, 8, 5] | [] | 5/5 ✓ | 0.1169 |
| 2 | [1, 48, 26, 31, 27] | [] | 5/5 ✓ | 0.1300 |
| 3 | [4, 29, 40, 12, 15] | [] | 5/5 ✓ | 0.1772 |
| 4 | [28, 11, 23, 35, 20] | [] | 5/5 ✓ | 0.1190 |
| 5 | [6, 45, 14, 25, 0] | [] | 5/5 ✓ | 0.1466 |
| 6 | [49] | [19, 43, 42, 39] | 5/5 ✓ | 0.1909 |
| 7 | [] | [32] | 1/5 ✓ | 0.0367 |

**Total Transaction Cost:** 0.9173

### Constraint Satisfaction

| Constraint | Required | Achieved | Status |
|------------|----------|----------|--------|
| Portfolio Size | 15 | 15 | ✅ |
| Max per Sector | ≤5 | 4 | ✅ |
| Max Changes/Week | ≤5 | 5 | ✅ |

---

## How to Run

### Prerequisites

```bash
pip install numpy scipy
```

### For Quantum Solution (optional)

```bash
pip install qiskit qiskit-optimization qiskit-algorithms
```

### Running the Classical Solution

```bash
python classical_solution_FINAL.py
```

### Output Files

| File | Description |
|------|-------------|
| `classical_results.json` | Complete results in JSON format |
| `classical_selected.npy` | Selected asset indices |
| `classical_weights.npy` | Optimal portfolio weights |

---

## File Structure

```
quantum_portfolio/
│
├── README.md                      # This file
│
├── data/
│   ├── expected_returns.npy       # Asset returns
│   ├── covariance_matrix.npy      # Risk matrix
│   ├── previous_positions.npy     # Current holdings
│   ├── transaction_costs.npy      # Trading fees
│   ├── sector_list.npy            # Sector labels
│   └── volatilities.npy           # Asset volatilities
│
├── classical/
│   ├── classical_solution_FINAL.py    # Classical solution
│   ├── classical_results.json         # Results
│   ├── classical_selected.npy         # Selected assets
│   └── classical_weights.npy          # Optimal weights
│
├── quantum/
│   ├── quantum_solution.py        # QAOA implementation
│   └── quantum_results.json       # Results
│
└── comparison/
    ├── comparison.py              # Compare classical vs quantum
    └── comparison_charts.png      # Visualization
```

---

## Key Learnings

### 1. Portfolio Optimization is Hard

With 50 assets and 15 to select, there are 2.25 trillion combinations. We can't check them all!

### 2. Constraints Make It Harder

Real-world constraints (sector limits, transaction limits) make the problem more complex but more realistic.

### 3. Two-Stage Approach Works Well

- Stage 1: Select assets (combinatorial - hard)
- Stage 2: Optimize weights (convex - easier)

### 4. Multi-Period Rebalancing is Practical

When you can't make all changes at once, spread them over time while respecting constraints.

### 5. Sharpe Ratio is King

Maximizing return is not enough. We need to maximize **risk-adjusted** return.

---

## Future Improvements

1. **Implement QAOA** on real quantum hardware
2. **Add more constraints** (liquidity, minimum position size)
3. **Real-time rebalancing** based on market conditions
4. **Backtest** on historical data
5. **Compare** multiple quantum algorithms (VQE, QAOA, Quantum Annealing)

---

## References

1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*
2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. *arXiv:1411.4028*
3. Qiskit Finance Tutorials: https://qiskit.org/documentation/finance/
4. IBM Quantum: https://quantum-computing.ibm.com/

---

## Authors

Gabriel Justina Ayomide

Africa Quantum Computing Hackathon 2024




