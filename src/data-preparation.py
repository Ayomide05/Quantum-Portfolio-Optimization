import pandas as pd
import numpy as np
import random


df = pd.read_excel('AQC Dataset.xlsx', engine='openpyxl')
n_assets = len(df)
df = df.rename(columns={'0Market_Cap (billions)': 'Market_Cap'})   # Rename Market Cap Column
df.columns = df.columns.str.strip() # Strip Spaces from all column names
#Strip Spaces from all text data in the rows using applymap to check every single cell. if it's a string (text), we strip it, if it's a number, we leave it alone.
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  

#print("New Column Names:", df.columns.tolist())
#print("\nFirst 5 rows:")
#print(df.head())
# Addressing Missing Covariance
#Estimating Volatilities from Market Cap
market_caps = df['Market_Cap'].values
print(f"Market Cap range: ${market_caps.min():.1f}B to ${market_caps.max():.1f}B")

#Estimate Volatility: smaller cap = higher volatility: We are using σ = base_vol + vol_range / (1 + log(market_cap)) formular
base_volatility = 0.20  # Minimum volatility (20% for large caps)
volatility_range = 0.40 # Additional volatility for small caps

# Normalize market cap using log scale
log_caps = np.log(market_caps + 1)  # The +1 is to avoid log 0
normalized_caps = (log_caps - log_caps.min()) / (log_caps.max() - log_caps.min())

# Higher normalized cap = lower volatility
volatilities = base_volatility + volatility_range * (1 - normalized_caps)

df['Estimated_Volatility'] = volatilities
print(f"\nEstimated volatility range: {volatilities.min()*100:.1f}% to {volatilities.max()*100:.1f}%")
# Data Verification
print("\nSample volatilities:")
sample = df[['Asset', 'Sector', 'Market_Cap', 'Expected_Return', 'Estimated_Volatility']].head(10)
for _, row in sample.iterrows():
    print(f" {row['Asset']}: Market Cap ${row['Market_Cap']:.1f}B -> Volatility {row['Estimated_Volatility']*100:.1f}%")

# Build Correlation Matrix using a correlation logic(Same Sector - correlation 0.6 (move together)), Different sectors - correlation 0.2 (some general market correlation)
sectors = df['Sector'].values
unique_sectors = df['Sector'].unique()
print(f"\nSectors: {list(unique_sectors)}")

correlation_matrix = np.zeros((n_assets, n_assets))

SAME_SECTOR_CORR = 0.60   #Correlation within the same sector
DIFF_SECTOR_CORR = 0.20   #Correlation between different sectors

for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            correlation_matrix[i, j] = 1.0  # Perfect correlation with self
        elif sectors[i] == sectors[j]:
            noise = random.uniform(-0.05, 0.05)
            correlation_matrix[i, j] = SAME_SECTOR_CORR + noise
        else:
            noise = random.uniform(-0.05, 0.05)
            correlation_matrix[i, j] = DIFF_SECTOR_CORR

# Build Covariance Matrix using Covariance σᵢⱼ = Correlation ρᵢⱼ × Volatility σᵢ × Volatility σⱼ formular
# Note: For diagonal (variance): σᵢᵢ = σᵢ² (since ρᵢᵢ = 1)

covariance_matrix = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(n_assets):
        covariance_matrix[i, j] = (correlation_matrix[i, j] * volatilities[i] * volatilities[j])


# To verify positive semi-definite (PSD) which is required for valid covariance matrix
eigenvalues = np.linalg.eigvalsh(covariance_matrix)
is_psd = np.all(eigenvalues >= -1e-10)
#print(f"\nMatrix is positive semi-definite: {is_psd}")

# Expected returns vector (μ)
expected_returns = df['Expected_Return'].values
print(f"Expected Returns (μ):")
print(f"  Shape: ({len(expected_returns)},)")
print(f"  Range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
print(f"  Mean: {expected_returns.mean():.2%}")

# Previous positions vector
previous_positions = df['Previous_Position'].values
print(f"\nPrevious Positions (x_prev):")
print(f"  Shape: ({len(previous_positions)},)")
print(f"  Currently holding: {int(previous_positions.sum())} assets")

# Transaction costs vector (per asset)
transaction_costs = df['Transaction_Cost'].values
print(f"\nTransaction Costs (τ):")
print(f"  Shape: ({len(transaction_costs)},)")
print(f"  Range: {transaction_costs.min():.2%} to {transaction_costs.max():.2%}")
print(f"  Mean: {transaction_costs.mean():.2%}")

# Sector mapping
sector_list = df['Sector'].tolist()
unique_sectors = list(df['Sector'].unique())
print(f"\nSectors:")
print(f"  Unique sectors: {unique_sectors}")
for sec in unique_sectors:
    count = sector_list.count(sec)
    print(f" {sec}: {count} assets")

# Optimization Parameters from problem statement
N = 15    # Target portfolio size (must hold exactly 15 assets)
K = 5     # Maximum position changes allowed
S_max = 0.35    # MAximujm sector allocation (35%)
lambda_risk  = 0.5   # Risk aversion parameter (λ)

print (f'Parameters from problem statement:')
print(f"    N (POrtfolio size):     {N} assets")
print(f"    K (max changes):        {K} trades")
print(f"    S_max (sector_limit):   {S_max:.0%} of portfolio = {int(S_max * N)} assets max per sector")
print(f"    λ (risk aversion):      {lambda_risk}")

# Current Situation
current_holdings = int(previous_positions.sum())
print(f"\nCurrent situation:")
print(f"    Currently holding:      {current_holdings} assets")
print(f"    Need to reach:          {N} assets")
print(f"    Difference:             {current_holdings - N} (need to sell {current_holdings - N})")


# We are curently holding 36 assets, The Target Portfolio is 15 assets and the maximum changes allowed is 5.
# To go from 36 assets to 15 assets, we need to sell 21 assets but K=5 only allows 5 changes which is impossible in one rebalancing period.

#save all our parameters to use for the next step
np.save('expected_returns.npy', expected_returns)
np.save('covariance_matrix.npy', covariance_matrix)
np.save('previous_positions.npy', previous_positions)
np.save('transaction_costs.npy', transaction_costs)
np.save('volatilities.npy', volatilities)

# Save Sector info as numpy array of indices
sector_indices = {}
for sec in unique_sectors:
    sector_indices[sec] = [i for i, s in enumerate(sector_list) if s == sec]
    
np.save('sector_list.npy', np.array(sector_list))
df.to_csv('prepared_dataset.csv', index=False)

# Save parameters
params = {
    'N': N,
    'K': K,
    'S_max': S_max,
    'lambda_risk': lambda_risk,
    'n_assets': n_assets,
    'sectors': unique_sectors,
    'current_holdings': current_holdings
}

import json
with open('parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

