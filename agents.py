"""
Agent-Based Model for Residential Segregation
----------------------------------------------
This script implements the agent-based model for residential segregation.

IMPORTANT: Each agent represents ONE HOUSEHOLD.

The agents (households) have the following attributes:
- Group identity (B=Black, W=White, A=Asian, H=Hispanic/Latino)
- Income (ranked 1-10, mapped to dollar amounts) - household income
- Education (years) - typically head of household education
- Location (tract assignment) - where the household resides

Agents are initialized to match the initial group shares (s_B0, s_W0, s_A0, s_H0)
from the tract data, respecting tract capacities (K = total housing units).
The number of agents (households) should not exceed the total capacity.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path


# ==============================
# Agent configuration and helpers
# ==============================

# Groups: 0=Black, 1=White, 2=Asian, 3=Hispanic
GROUPS = np.array(["B", "W", "A", "H"])
G = len(GROUPS)

# Income ranks 1..10
R = 10

# Income midpoints by rank (calibrated to Chicago income distribution)
# Based on ACS median household income ranges
# Ranks roughly correspond to income deciles
Y_MIDPOINTS = np.array([
    15_000,   # Rank 1: <$20k
    30_000,   # Rank 2: $20k-$35k
    45_000,   # Rank 3: $35k-$50k
    60_000,   # Rank 4: $50k-$65k
    75_000,   # Rank 5: $65k-$80k
    95_000,   # Rank 6: $80k-$100k
    125_000,  # Rank 7: $100k-$125k
    150_000,  # Rank 8: $125k-$150k
    175_000,  # Rank 9: $150k-$200k
    250_000   # Rank 10: >$200k
])


def income_from_rank(rank_array: np.ndarray) -> np.ndarray:
    """
    Map income ranks (1..R) to midpoints in dollars.
    rank_array can be scalar or 1D array of ints.
    """
    return Y_MIDPOINTS[rank_array - 1]


def initialize_agents(
    tracts: gpd.GeoDataFrame,
    N_households: int | None = None,
    use_tract_shares: bool = True,
    occupancy_rate: float = 0.95,
    rank_probs: np.ndarray | None = None,
    edu_mean: float = 13.5,
    edu_sd: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create the agent (household) population and assign them to tracts.
    
    NOTE: Each agent = 1 household. The number of agents should not exceed
    the total housing capacity (sum of K across all tracts).

    Parameters
    ----------
    tracts : GeoDataFrame
        Must have columns: 'tract_id', 'K', 's_B0', 's_W0', 's_A0', 's_H0'.
        These are the initial group shares from prepare_tract.py.
        K = total housing units (capacity) in each tract.
    N_households : int, optional
        Number of agents/households in the simulation.
        If None, uses sum(K) * occupancy_rate to match actual occupancy.
        Should not exceed sum(K) across all tracts.
    use_tract_shares : bool, default True
        If True, uses tract-level initial group shares (s_B0, s_W0, s_A0, s_H0) to assign
        agents (households) to groups based on their initial tract location.
        If False, uses city-wide group probabilities.
    occupancy_rate : float, default 0.95
        Fraction of capacity to fill with agents (if N_households is None).
        Represents that not all housing units are occupied (typical occupancy ~95%).
    rank_probs : array-like of length R, optional
        Probabilities for income ranks 1..R. If None, uses a realistic distribution.
        Income represents household income.
    edu_mean : float
        Mean years of education (typically head of household).
    edu_sd : float
        Standard deviation of years of education.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    agents : DataFrame
        Each row represents ONE HOUSEHOLD (agent).
        Columns:
            - agent_id: Unique identifier for the household
            - group_idx: Group index (0=B, 1=W, 2=A, 3=H)
            - group: Group label ("B", "W", "A", "H")
            - rank: Income rank (1-10) - household income rank
            - income: Income in dollars - household income
            - edu: Years of education - typically head of household
            - tract_idx: Index into tracts GeoDataFrame
            - tract_id: Tract identifier (GEOID) - where household resides
    """
    rng = np.random.default_rng(seed)
    
    # Validate required columns
    required_cols = ['tract_id', 'K', 's_B0', 's_W0', 's_A0', 's_H0']
    missing_cols = [col for col in required_cols if col not in tracts.columns]
    if missing_cols:
        raise ValueError(f"tracts must have columns: {required_cols}. Missing: {missing_cols}")
    
    tract_ids = tracts["tract_id"].values
    K = tracts["K"].values.astype(float)
    s_B0 = tracts["s_B0"].values
    s_W0 = tracts["s_W0"].values
    s_A0 = tracts["s_A0"].values
    s_H0 = tracts["s_H0"].values
    N_tr = len(tracts)
    
    if (K <= 0).any():
        raise ValueError("Some tracts have non-positive capacity K.")
    
    # Determine number of households (agents)
    total_capacity = np.sum(K)
    if N_households is None:
        # Use total capacity * occupancy rate
        N_households = int(total_capacity * occupancy_rate)
        print(f"Initializing {N_households} households (agents) from {total_capacity:.0f} total capacity (occupancy rate: {occupancy_rate:.1%})")
    else:
        # Validate that we don't exceed capacity
        if N_households > total_capacity:
            print(f"Warning: N_households ({N_households}) exceeds total capacity ({total_capacity:.0f})")
            print(f"  Capping at total capacity.")
            N_households = int(total_capacity)
    
    # 1. Assign tracts to agents (initial location) - proportional to capacity
    tract_probs = K / K.sum()
    tract_idx = rng.choice(np.arange(N_tr), size=N_households, p=tract_probs)
    tract_id = tract_ids[tract_idx]
    
    # 2. Assign groups to agents based on tract-level initial shares
    # IMPORTANT: Each agent (household) is assigned to exactly ONE group (mutually exclusive)
    if use_tract_shares:
        # For each agent, assign group based on their tract's initial group shares
        # The shares (s_B0, s_W0, s_A0, s_H0) are mutually exclusive and sum to 1.0
        group_idx = np.zeros(N_households, dtype=int)
        for i in range(N_households):
            tr_idx = tract_idx[i]
            # Get group probabilities for this tract (4 mutually exclusive groups)
            tract_group_probs = np.array([s_B0[tr_idx], s_W0[tr_idx], s_A0[tr_idx], s_H0[tr_idx]])
            # Normalize to ensure they sum to 1 (should already be normalized, but safety check)
            tract_group_probs = tract_group_probs / (tract_group_probs.sum() + 1e-10)
            # Assign agent to exactly one group
            group_idx[i] = rng.choice(np.arange(G), p=tract_group_probs)
    else:
        # Use city-wide group probabilities (weighted by capacity)
        city_group_probs = np.array([
            np.sum(s_B0 * K) / np.sum(K),
            np.sum(s_W0 * K) / np.sum(K),
            np.sum(s_A0 * K) / np.sum(K),
            np.sum(s_H0 * K) / np.sum(K)
        ])
        city_group_probs = city_group_probs / city_group_probs.sum()
        group_idx = rng.choice(np.arange(G), size=N_households, p=city_group_probs)
    
    group = GROUPS[group_idx]
    
    # 3. Income rank distribution - DATA-DRIVEN based on tract median income
    # Check for median_income column or census variable B19013_001
    income_col = None
    if 'median_income' in tracts.columns:
        income_col = 'median_income'
    elif 'B19013_001' in tracts.columns:
        income_col = 'B19013_001'
    
    if income_col is not None:
        # Use tract-level median income to determine income distribution
        median_income = tracts[income_col].values
        
        # Handle census special values (missing data codes)
        census_na_values = [-666666666, -999999999, -222222222]
        median_income = pd.Series(median_income).replace(census_na_values, np.nan)
        
        # Convert to numeric and handle missing values - use median of all tracts
        median_income = pd.to_numeric(median_income, errors='coerce')
        median_income_filled = median_income.fillna(median_income.median())
        median_income = median_income_filled.values
        
        # Map tract median income to income rank probabilities
        # Higher median income tracts -> higher income ranks on average
        ranks = np.zeros(N_households, dtype=int)
        
        for i in range(N_households):
            tr_idx = tract_idx[i]
            tract_median = median_income[tr_idx]
            
            # Create income rank probabilities based on tract median income
            # Map median income to a target rank (1-10 scale)
            # Use income midpoints as reference points
            target_rank = np.searchsorted(Y_MIDPOINTS, tract_median, side='right')
            target_rank = np.clip(target_rank, 1, R)  # Ensure in range [1, R]
            
            # Add random variation to target rank (so agents in same tract have different targets)
            # This ensures variation even within the same tract
            target_rank_varied = target_rank + rng.integers(-2, 3)  # Add -2 to +2 random variation
            target_rank_varied = np.clip(target_rank_varied, 1, R)
            
            # Create distribution centered around varied target rank
            # Higher median income -> distribution shifted toward higher ranks
            rank_probs_tract = np.zeros(R)
            for r in range(1, R + 1):
                # Distance from varied target rank
                dist = abs(r - target_rank_varied)
                # Probability decreases with distance (wider spread for more variation)
                rank_probs_tract[r - 1] = np.exp(-0.5 * (dist / 2.5) ** 2)  # Wider spread (2.5 instead of 1.5)
            
            # Add significant uniform component for more variation
            # Mix with uniform to ensure agents in same tract have different incomes
            uniform_component = np.ones(R) / R
            rank_probs_tract = 0.5 * rank_probs_tract + 0.5 * uniform_component  # 50/50 mix for more variation
            rank_probs_tract = rank_probs_tract / rank_probs_tract.sum()
            
            # Sample rank for this agent (independent sampling ensures variation)
            ranks[i] = rng.choice(np.arange(1, R + 1), p=rank_probs_tract)
        
        income = income_from_rank(ranks)
        print(f"  Income distribution: data-driven from tract median income (column: {income_col})")
    else:
        # Fallback to fixed distribution if median_income not available
        if rank_probs is None:
            # Realistic income distribution (skewed toward lower/middle incomes)
            # Based on typical US income distribution
            rank_probs = np.array([
                0.15,  # Rank 1: <$20k
                0.14,  # Rank 2: $20k-$35k
                0.13,  # Rank 3: $35k-$50k
                0.12,  # Rank 4: $50k-$65k
                0.11,  # Rank 5: $65k-$80k
                0.10,  # Rank 6: $80k-$100k
                0.09,  # Rank 7: $100k-$125k
                0.08,  # Rank 8: $125k-$150k
                0.05,  # Rank 9: $150k-$200k
                0.03   # Rank 10: >$200k
            ])
        rank_probs = np.array(rank_probs, dtype=float)
        rank_probs = rank_probs / rank_probs.sum()
        
        ranks = rng.choice(np.arange(1, R + 1), size=N_households, p=rank_probs)
        income = income_from_rank(ranks)
        print(f"  Income distribution: fixed (median_income not found in tracts)")
    
    # 4. Education distribution (normal, truncated)
    edu = rng.normal(loc=edu_mean, scale=edu_sd, size=N_households)
    edu = np.clip(edu, 8.0, 20.0)  # clamp to [8, 20] years
    
    # 5. Build DataFrame
    # Each row = 1 agent = 1 household
    agents = pd.DataFrame({
        "agent_id": np.arange(N_households),
        "group_idx": group_idx,
        "group": group,
        "rank": ranks,
        "income": income,  # Household income
        "edu": edu,  # Head of household education
        "tract_idx": tract_idx,
        "tract_id": tract_id,
    })
    
    return agents


# ==============================
# Example usage
# ==============================

if __name__ == "__main__":
    # Load tracts from prepared data
    tracts_path = Path("data") / "chicago_tracts_enriched.shp"
    
    if not tracts_path.exists():
        print(f"Error: Tracts file not found at {tracts_path}")
        print("Please run prepare_tract.py first to generate the enriched tracts.")
    else:
        print(f"Loading tracts from {tracts_path}...")
        tracts = gpd.read_file(tracts_path)
        
        # Initialize agents
        # Use occupancy_rate to match realistic occupancy (not all units are filled)
        agents = initialize_agents(
            tracts,
            N_households=None,  # Will use sum(K) * occupancy_rate
            use_tract_shares=True,  # Use tract-level initial group shares
            occupancy_rate=0.95,
            seed=42
        )
        
        print(f"\nInitialized {len(agents)} agents (households)")
        print(f"Total capacity: {tracts['K'].sum():.0f} housing units")
        print(f"Occupancy: {len(agents) / tracts['K'].sum():.1%}")
        print(f"\nGroup distribution (households):")
        print(agents['group'].value_counts().sort_index())
        print(f"\nGroup proportions:")
        print(agents['group'].value_counts(normalize=True).sort_index())
        print(f"\nHousehold income statistics:")
        print(agents['income'].describe())
        print(f"\nFirst few households (agents):")
        print(agents.head(10))
