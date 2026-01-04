# Quantum Portfolio Optimization 🌍
### Africa Quantum Computing Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![D-Wave](https://img.shields.io/badge/Quantum-D--Wave-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A comprehensive portfolio optimization solution using both **Classical** (Simulated Annealing) and **Quantum** (Quantum Annealing) approaches to solve a real-world financial optimization problem.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Mathematical Background](#-mathematical-background)
- [Data Preparation](#-data-preparation)
- [Solution Approach](#-solution-approach)
- [Innovations](#-innovations-key-differentiators)
- [Results & Comparison](#-results--comparison)
- [How to Run](#-how-to-run)
- [File Structure](#-file-structure)
- [References](#-references)

---

## 🎯 Problem Statement

**The Challenge:**
As a quantum finance analyst at the Africa Quantum Consortium, the goal is to rebalance a portfolio of **50 assets**. Due to transaction costs and market impact, we must optimize the selection under strict constraints.

**Objective:**
Find the optimal set of assets to buy/sell/hold that **maximizes expected returns** while:
1. Staying within risk tolerance limits.
2. Minimizing transaction costs.
3. Maintaining sector diversification.
4. Limiting position changes to **K** assets per period.

### Constraints
| Constraint | Description | Value |
| :--- | :--- | :--- |
| **Portfolio Size (N)** | Number of assets to hold | **15** |
| **Sector Limit (S_max)** | Maximum assets per sector | **5** |
| **Change Limit (K)** | Max position changes per week | **5** |
| **Transaction Costs** | Fees for buying/selling | **Minimize** |

---

## 🧮 Mathematical Background

### 1. Portfolio Metrics
**Portfolio Return:**
$$Return = \sum (w_i \times \mu_i)$$
*Where $w_i$ is the weight and $\mu_i$ is the expected return.*

**Portfolio Risk (Variance):**
$$\sigma^2_p = w^T \Sigma w$$
*Where $\Sigma$ is the Covariance Matrix.*

**Sharpe Ratio (The Objective):**
$$\text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}$$
*Higher is better.*

### 2. QUBO Formulation
We treat asset selection as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

$$\text{Minimize: } -\sum (\mu_i x_i) + \lambda \sum (\sigma_{ij} x_i x_j) + \tau \sum |x_i - x_{prev}|$$

**Subject to:**
* $\sum x_i = N$ (Portfolio Size)
* $\sum_{sector} x_i \le S_{max}$ (Sector Limit)

---

## 📂 Data Preparation

The original dataset provided returns and sector info but lacked a historical covariance matrix. We derived a robust risk model using financial principles.

### 1. Volatility Estimation
We used the **Market Cap to Volatility** relationship (small caps tend to be more volatile).
* **Formula:** `volatility = base + range * (1 - normalized_market_cap)`
* **Result:**
    * **Large Cap:** ~20% Volatility (Lower Risk)
    * **Small Cap:** ~60% Volatility (Higher Risk)

### 2. Correlation Matrix
We constructed correlations based on sector membership:
* **Same Asset:** $\rho = 1.0$
* **Same Sector:** $\rho = 0.6$ (Assets in the same industry move together).
* **Different Sector:** $\rho = 0.2$ (General market correlation).

### 3. Covariance Calculation
The final Covariance Matrix $\Sigma$ was computed as:
$$\Sigma_{ij} = \rho_{ij} \times \sigma_i \times \sigma_j$$

* **Validation:** The resulting matrix is positive semi-definite and symmetric, ensuring valid optimization.

---

## 💡 Solution Approach

We employed a **Two-Stage Optimization** strategy:

1.  **Stage 1: Asset Selection (Combinatorial)**
    * **Classical:** Simulated Annealing.
    * **Quantum:** D-Wave Quantum Annealing (via `neal`).
    * *Goal:* Select the best 15 assets out of 50.

2.  **Stage 2: Weight Allocation (Continuous)**
    * **Method:** Scipy SLSQP (Sequential Least Squares Programming).
    * *Goal:* Assign optimal weights to the 15 selected assets to maximize the Sharpe Ratio.

---

## 🚀 Innovations (Key Differentiators)

### Innovation 1: Hybrid Quantum-Classical Warm Start
Instead of starting the quantum annealing process from random noise, we injected "intelligent guesses" based on classical heuristics.

| Method | Strategy | Result |
| :--- | :--- | :--- |
| **Greedy Sharpe** | Pick top assets by Sharpe Ratio | High Return |
| **Favor Holdings** | Keep current positions to save fees | **Winner (Best Balance)** |
| **Sector Balanced** | Force equal sector distribution | Low Risk |

**Result:** The "Favor Holdings" warm start achieved the highest efficiency, balancing returns with transaction costs.

### Innovation 2: Real-Time Rebalancing Trigger
We implemented a **Mahalanobis Distance Trigger** to monitor portfolio drift. Unlike Euclidean distance, Mahalanobis accounts for asset correlations.

$$d(w, w^*) = \sqrt{(w - w^*)^T \Sigma^{-1} (w - w^*)}$$

* **Logic:** Only rebalance if the "Risk-Adjusted Drift" exceeds a threshold.
* **30-Day Simulation Result:** The portfolio remained stable (Drift < 0.25), resulting in **Zero Unnecessary Trades**.

---

## 📊 Results & Comparison

| Metric | Classical SA | Quantum QA | Improvement |
| :--- | :--- | :--- | :--- |
| **Expected Return** | 69.69% | **101.33%** | 🔼 **+31.64%** |
| **Portfolio Risk** | 23.90% | **20.50%** | 🔽 **-3.40%** |
| **Sharpe Ratio** | 2.83 | **4.85** | 🚀 **+71.4%** |

### Quantum Portfolio Allocation
* **Selected Assets:** `[2, 17, 18, 19, 21, 25, 32, 35, 36, 39, 40, 42, 43, 47, 48]`
* **Top Sector:** Energy (34.89%) & Consumer (32.58%)
* **Constraint Check:**
    * Size: 15 ✅
    * Sector Limit: Max 3 ✅
    * Turnover: Max 5 changes/week ✅

---

## 💻 How to Run

### 1. Prerequisites
```bash
pip install numpy scipy matplotlib pandas openpyxl dwave-neal dimod

---
### 2. Data Preparation
```bash
cd data
python data_preparation.py
---
### 3. Run Classical Baseline
```bash
cd ../src
python classical_optimization.py
---
### 4. Run Quantum Optimization
```bash
python quantum_optimization.py
---
### 5. Generate Visualizations
```bash
python visualize_results.py

## File Structure
```Plaintext
quantum_portfolio/
│
├── README.md                       # Project Documentation
├── data/
│   ├── data_preparation.py         # Preprocessing script
│   ├── expected_returns.npy        # Generated inputs
│   ├── covariance_matrix.npy
│   └── ...
│
├── src/
│   ├── classical_optimization.py   # Simulated Annealing
│   ├── quantum_optimization.py     # D-Wave Quantum Annealing
│   └── visualize_results.py        # Chart generation
│
└── figures/                        # Generated output charts
    ├── performance_comparison.png
    ├── sector_allocation.png
    └── innovation_summary.png

---

## References

1. Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.

2. D-Wave Systems. Ocean Documentation.

3. Venturelli, D., et al. (2019). Reverse quantum annealing approach to portfolio optimization.

4. Farhi, E., et al. (2014). A Quantum Approximate Optimization Algorithm

---

## Authors

Gabriel Justina Ayomide

Africa Quantum Computing Hackathon 2025




