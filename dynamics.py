"""
Agent Relocation Dynamics Module
--------------------------------
This module implements the relocation dynamics for the agent-based segregation model.

The dynamics determine how agents (households) decide to move between tracts
based on various factors such as:
- Housing costs (rent)
- Amenity values
- Commuting distance
- Group preferences
- Capacity constraints
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Import constants and functions from agents module
from agents import GROUPS, G, R, income_from_rank

# =========================================
# DYNAMICS AND MODEL UPDATES
# =========================================

# ---------- 1. Parameters ----------

# Intergenerational income mobility parameters (research-based on Chetty et al. and related mobility literature
INCOME_PERSISTENCE = 0.45  # Correlation between parent and child income (0.4-0.5 in research)
NEIGHBORHOOD_EFFECT_STRENGTH = 0.5  # Impact of neighborhood quality on income mobility
EDUCATION_PREMIUM = 0.15  # Rank increase per year of education beyond high school (12 years)
INCOME_SHOCK_SD = 0.8  # Standard deviation of random income shocks

# Group-specific mobility barriers (structural inequality)
# Negative values = barriers to upward mobility, positive = advantages
GROUP_MOBILITY_EFFECTS = {
    "B": -0.4,  # Black: significant structural barriers
    "H": -0.2,  # Hispanic: moderate barriers
    "A": -0.1,  # Asian: slight barrier
    "W": 0.0,  # White: baseline (no penalty/advantage)
}

# Fraction of people within each race that can experience income mobility each period
# Varies by race - different groups have different rates of income mobility
mu_income_mobility_by_group = {
    "B": 0.25,  # Black: 25% can experience income mobility
    "H": 0.30,  # Hispanic: 30% can experience income mobility
    "A": 0.35,  # Asian: 35% can experience income mobility
    "W": 0.40,  # White: 40% can experience income mobility (highest)
}

# Rent dynamics parameters
lambda_rent = 0.3
u_bar = 0.85
alpha_rent = 1.5
beta_rent = 0.8

# Affordability thresholds
theta_aff = 0.3     # tract rent must be <= 0.3 * income to be in feasible set
gamma_priceout = 0.4  # forced move if current rent >= 0.4 * income

# Utility parameters
alpha_social = 5.0  # weight on group composition preferences (increased to promote segregation)
alpha_d = 0.05   # commuting disutility weight (very small so social preferences dominate)
alpha_rent_util = 0.01  # rent disutility weight in utility (normalized - rent values are large)
k_move = 1.0    # fixed moving cost (reduced to allow more movement for segregation)
beta_logit = 2.0  # logit precision (not used when moving to best feasible tract)
n_candidates = 10  # number of candidate tracts to consider (increased for better search)

# Group-specific fractions that decide to move each period
mu_active_by_group = {
    "B": 0.3,  # fraction of Black households that move each period
    "W": 0.3,  # fraction of White households that move each period
    "A": 0.3,  # fraction of Asian households that move each period
    "H": 0.3,  # fraction of Hispanic households that move each period
}

# Bidirectional stereotype matrix B[g, h]
# order of groups: B, W, A, H (must match GROUPS)
# Bmat[i, j] = preference of group i for group j
# Increased outgroup aversion (more negative values) to promote stronger segregation
Bmat = np.array([
    [ 1.0, -1.0,  0.0, -0.8],   # preferences of B toward (B, W, A, H) - stronger aversion to W and H
    [-1.2,  1.0,  0.2, -0.8],   # preferences of W toward (B, W, A, H) - stronger aversion to B and H
    [-0.3,  0.1,  1.0, -0.2],   # preferences of A toward (B, W, A, H) - slight aversion to B and H
    [-0.5, -0.8, -0.2,  1.0],   # preferences of H toward (B, W, A, H) - stronger aversion to W, slight to B and A
])


# ---------- 2. Helper: occupancy and shares ----------

def compute_tract_occupancy(agents: pd.DataFrame, tracts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given agents with 'tract_idx' and 'group_idx', compute:
      H_n  : (N_tr,) total households per tract
      H_ng : (N_tr, G) households per (tract, group)
      s_ng : (N_tr, G) shares per (tract, group)
    """
    N_tr = len(tracts)
    H_n = np.bincount(agents["tract_idx"].values, minlength=N_tr).astype(float)

    H_ng = np.zeros((N_tr, G), dtype=float)
    for g_idx in range(G):
        mask = agents["group_idx"].values == g_idx
        if mask.sum() == 0:
            continue
        tr_counts = np.bincount(agents.loc[mask, "tract_idx"].values, minlength=N_tr)
        H_ng[:, g_idx] = tr_counts

    s_ng = np.zeros_like(H_ng)
    positive = H_n > 0
    if positive.any():
        s_ng[positive, :] = H_ng[positive, :] / H_n[positive, None]
    
    # Ensure no NaN or inf values
    s_ng = np.nan_to_num(s_ng, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to ensure shares sum to 1 for occupied tracts
    row_sums = s_ng.sum(axis=1)
    positive_sums = row_sums > 0
    if positive_sums.any():
        s_ng[positive_sums, :] = s_ng[positive_sums, :] / row_sums[positive_sums, None]

    return H_n, H_ng, s_ng


# ---------- 3. Rent update ----------

def rent_update_step(omega_t: np.ndarray, tracts, H_n: np.ndarray) -> np.ndarray:
    """
    One-period update for tract rents ω_{n,t+1}.
    
    Note: Uses column names from prepare_tract.py:
    - 'p0' for baseline rent (omega_bar)
    - 'amenity' for amenity index (v_n)
    """
    K = tracts["K"].values.astype(float)
    
    # Use 'p0' if available, otherwise try 'omega_bar'
    if "p0" in tracts.columns:
        omega_bar = tracts["p0"].values
    elif "omega_bar" in tracts.columns:
        omega_bar = tracts["omega_bar"].values
    else:
        raise ValueError("Tracts must have 'p0' or 'omega_bar' column")
    
    # Use 'amenity' if available, otherwise try 'v_n'
    if "amenity" in tracts.columns:
        v_n = tracts["amenity"].values
    elif "v_n" in tracts.columns:
        v_n = tracts["v_n"].values
    else:
        raise ValueError("Tracts must have 'amenity' or 'v_n' column")

    u_n = np.where(K > 0, H_n / K, 0.0)

    omega_star = omega_bar * (1 + alpha_rent * (u_n - u_bar) + beta_rent * v_n)
    omega_star = np.maximum(omega_star, 200.0)  # keep rents positive

    omega_next = (1 - lambda_rent) * omega_t + lambda_rent * omega_star
    return omega_next


# ---------- 4. Moving motivation ----------

def motivated_to_move(income: np.ndarray, current_rent: np.ndarray) -> np.ndarray:
    """
    Return status (array of ints):
      0 = not motivated (if we ignore for this period)
      1 = wants upgrade (can afford better)
      2 = priced out (forced mover)
    """
    status = np.zeros_like(income, dtype=int)

    phi_upgrade = 0.15  # lower threshold, "I can afford something better"

    affordable = current_rent <= phi_upgrade * income
    priced_out = current_rent >= gamma_priceout * income

    status[affordable] = 1
    status[priced_out] = 2
    return status


# ---------- 5. Utility over tracts for a single agent ----------

def tract_utilities_for_agent(
    agent_idx: int,
    agents: pd.DataFrame,
    tracts,
    omega_t: np.ndarray,
    s_ng: np.ndarray,
) -> np.ndarray:
    """
    Compute deterministic part of utility U_i(n) for agent i over all tracts n:
        alpha_social * (B[g,:] · s_n)  (weighted group composition preferences)
      - alpha_d * dw_n  (distance enters utility directly)
      - alpha_rent_util * omega_n  (rent enters utility directly, normalized)
      - k_move if n != current tract (moving cost)
    """
    N_tr = len(tracts)

    g_i = agents.loc[agent_idx, "group_idx"]
    current_t = agents.loc[agent_idx, "tract_idx"]

    beta_vec = Bmat[g_i, :]  # (G,)
    
    # Handle NaN values in s_ng (empty tracts)
    s_ng_clean = np.nan_to_num(s_ng, nan=0.0, posinf=0.0, neginf=0.0)
    social_term = s_ng_clean @ beta_vec  # (N_tr,)
    
    # Replace any NaN or inf values with 0
    social_term = np.nan_to_num(social_term, nan=0.0, posinf=0.0, neginf=0.0)

    d_w = tracts["dw"].values
    
    # Rent enters utility directly (as disutility)
    # Use small weight since rent values are typically large (hundreds/thousands)
    rent_term = alpha_rent_util * omega_t

    # Weight social preferences more heavily to promote segregation
    util = alpha_social * social_term - alpha_d * d_w - rent_term

    # moving cost (increased so staying put is often optimal)
    move_penalty = np.zeros(N_tr)
    move_penalty[np.arange(N_tr) != current_t] = k_move
    util = util - move_penalty
    
    # Final cleanup
    util = np.nan_to_num(util, nan=-1e6, posinf=1e6, neginf=-1e6)

    return util


# ---------- 6. Income dynamics ----------

def income_transition_step(
    agents: pd.DataFrame,
    theta_rank: np.ndarray,
    kappa_rank: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Realistic income transition based on:
    1. Intergenerational persistence (research: ~0.4-0.5 correlation)
    2. Neighborhood effects (Chetty et al.: neighborhood quality affects mobility)
    3. Education returns (each year beyond high school increases income)
    4. Group-specific mobility barriers (structural inequality)
    5. Random economic shocks (business cycles, luck)
    
    Only a fraction (mu_income_mobility_by_group) of each group can experience
    income mobility each period. The fraction varies by race:
    - White: 40% can experience mobility
    - Asian: 35% can experience mobility
    - Hispanic: 30% can experience mobility
    - Black: 25% can experience mobility
    The rest stay at their current rank.
    """
    ranks = agents["rank"].values.copy()
    edu = agents["edu"].values
    group_arr = agents["group"].values
    tract_idx = agents["tract_idx"].values

    # Get tract characteristics (neighborhood quality)
    if "amenity" in tracts.columns:
        tract_amenity = tracts["amenity"].values[tract_idx]
    elif "v_n" in tracts.columns:
        tract_amenity = tracts["v_n"].values[tract_idx]
    else:
        tract_amenity = np.zeros(len(agents))

    new_ranks = ranks.copy()

    for g_char in GROUPS:
        mask = (group_arr == g_char)
        if mask.sum() == 0:
            continue

        ranks_g = ranks[mask]
        edu_g = edu[mask]
        tract_amenity_g = tract_amenity[mask]

        # Only a fraction of this group can experience income mobility (varies by race)
        mu_g = mu_income_mobility_by_group.get(g_char, 0.3)
        can_move = rng.random(len(ranks_g)) < mu_g
        
        # Agents who cannot move stay at current rank
        ranks_new_g = ranks_g.copy()
        
        if can_move.sum() > 0:
            # Only process agents who can move
            ranks_movable = ranks_g[can_move]
            edu_movable = edu_g[can_move]
            amenity_movable = tract_amenity_g[can_move]

            # 1. INTERGENERATIONAL PERSISTENCE
            # Expected rank = persistence * current_rank + (1 - persistence) * mean_rank
            # Mean reversion: children of high-income parents tend to have lower income (and vice versa)
            mean_rank = (R + 1) / 2  # Middle rank (5.5 for R=10)
            expected_rank = INCOME_PERSISTENCE * ranks_movable + (1 - INCOME_PERSISTENCE) * mean_rank

            # 2. NEIGHBORHOOD EFFECT (Chetty et al.)
            # Better neighborhoods increase upward mobility
            # tract_amenity is in [-0.5, 0.5], scale to meaningful rank changes
            neighborhood_boost = NEIGHBORHOOD_EFFECT_STRENGTH * amenity_movable

            # 3. EDUCATION PREMIUM
            # Each year of education beyond high school (12 years) increases expected rank
            education_boost = EDUCATION_PREMIUM * (edu_movable - 12.0)

            # 4. GROUP-SPECIFIC MOBILITY BARRIERS
            # Structural inequality: some groups face barriers to upward mobility
            group_effect = GROUP_MOBILITY_EFFECTS.get(g_char, 0.0)

            # 5. RANDOM ECONOMIC SHOCKS
            # Business cycles, personal luck, health shocks, etc.
            shocks = rng.normal(0, INCOME_SHOCK_SD, size=len(ranks_movable))
 
            # COMBINE ALL EFFECTS
            new_rank_float = (expected_rank +
                              neighborhood_boost +
                              education_boost +
                              group_effect +
                              shocks)

            # Discretize and bound to [1, R]
            ranks_movable_new = np.clip(np.round(new_rank_float), 1, R).astype(int)

            # Update only the movable agents
            ranks_new_g[can_move] = ranks_movable_new

        new_ranks[mask] = ranks_new_g

    agents["rank"] = new_ranks
    agents["income"] = income_from_rank(new_ranks)
    return agents

# ---------- 7. One-period update (full dynamics) ----------

def one_period_update(
    agents: pd.DataFrame,
    tracts,
    omega_t: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one period of the model:
      1) Income transition
      2) Decide active movers (group-specific fractions)
      3) Choose new tracts for active agents (local search: best from small candidate set)
         - Rents and distance enter utility directly
         - Agents only consider n_candidates tracts (local search)
         - Moving cost is high so staying put is often optimal
      4) Update occupancy and rents

    Returns
    -------
    agents      : updated DataFrame
    omega_next  : updated rents (array of length N_tr)
    H_n         : total households per tract
    H_ng        : group-specific counts per tract
    s_ng        : group shares per tract
    """
    N_tr = len(tracts)
    K = tracts["K"].values.astype(float)

    # 1) Income dynamics
    agents = income_transition_step(agents, tracts, rng)

    # 2) Occupancy + shares at start of period (before moves)
    H_n, H_ng, s_ng = compute_tract_occupancy(agents, tracts)

    # 3) Decide who is active and motivated (group-specific fractions)
    current_income = agents["income"].values
    current_rent = omega_t[agents["tract_idx"].values]

    move_status = motivated_to_move(current_income, current_rent)
    
    # Group-specific activity: only a fraction of each group are "shopping" this period
    active_flag = np.zeros(len(agents), dtype=bool)
    for group in GROUPS:
        group_mask = agents["group"].values == group
        group_indices = np.where(group_mask)[0]
        if len(group_indices) > 0:
            mu_g = mu_active_by_group.get(group, 0.3)
            group_active = (rng.random(len(group_indices)) < mu_g) & (move_status[group_indices] > 0)
            active_flag[group_indices] = group_active

    active_indices = np.where(active_flag)[0]

    # 4) For each active agent, choose the BEST feasible tract from candidate set (local search)
    for i in active_indices:
        y_i = current_income[i]
        current_t = agents.loc[i, "tract_idx"]

        # LOCAL SEARCH: Select a small candidate set of tracts to consider
        # Always include current tract (so staying put is an option)
        all_tracts = np.arange(N_tr)
        other_tracts = all_tracts[all_tracts != current_t]
        
        # Randomly sample n_candidates-1 other tracts (or all if fewer available)
        n_others = min(n_candidates - 1, len(other_tracts))
        if n_others > 0:
            candidate_others = rng.choice(other_tracts, size=n_others, replace=False)
            candidate_set = np.concatenate([[current_t], candidate_others])
        else:
            candidate_set = np.array([current_t])
        
        # Feasibility: rent <= θ_aff * y_i and tract not over capacity
        affordable = omega_t <= theta_aff * y_i
        has_space = H_n < K
        feasible = affordable & has_space

        # Only consider feasible tracts in candidate set
        candidate_feasible = candidate_set[feasible[candidate_set]]
        
        if len(candidate_feasible) == 0:
            continue  # no feasible tracts in candidate set

        # Compute utility only for candidate tracts
        util_det = tract_utilities_for_agent(i, agents, tracts, omega_t, s_ng)
        util_candidates = util_det[candidate_feasible]
        
        # Select the best tract from feasible candidates
        best_idx_in_candidates = np.argmax(util_candidates)
        new_t = candidate_feasible[best_idx_in_candidates]

        if new_t != current_t:
            # Update occupancy online (approx)
            H_n[current_t] -= 1
            H_n[new_t] += 1
            agents.at[i, "tract_idx"] = new_t
            agents.at[i, "tract_id"] = tracts["tract_id"].iloc[new_t]

    # 5) Recompute occupancy + shares after moves
    H_n, H_ng, s_ng = compute_tract_occupancy(agents, tracts)

    # 6) Update rents
    omega_next = rent_update_step(omega_t, tracts, H_n)

    return agents, omega_next, H_n, H_ng, s_ng


# ---------- 8. Run the full simulation ----------

def run_simulation(
    tracts,
    agents: pd.DataFrame,
    T: int = 20,
    seed: int = 0,
):
    """
    Run the model for T periods.

    Returns
    -------
    agents_final : DataFrame
    omega_path   : list of rent vectors (length T+1, including t=0)
    history      : list of dicts with snapshot info each period
                   (H_n, H_ng, s_ng)
    """
    rng = np.random.default_rng(seed)

    # initial rents = baseline rent (p0 or omega_bar)
    if "p0" in tracts.columns:
        omega_t = tracts["p0"].values.copy()
    elif "omega_bar" in tracts.columns:
        omega_t = tracts["omega_bar"].values.copy()
    else:
        raise ValueError("Tracts must have 'p0' or 'omega_bar' column")
    
    omega_path = [omega_t.copy()]
    history = []

    # initial snapshot
    H_n0, H_ng0, s_ng0 = compute_tract_occupancy(agents, tracts)
    history.append({"H_n": H_n0, "H_ng": H_ng0, "s_ng": s_ng0})

    for t in range(T):
        print(f"Period {t+1}/{T}")
        agents, omega_t, H_n, H_ng, s_ng = one_period_update(
            agents, tracts, omega_t, rng
        )
        omega_path.append(omega_t.copy())
        history.append({"H_n": H_n, "H_ng": H_ng, "s_ng": s_ng})

    return agents, omega_path, history


# ---------- 9. Example usage (you can comment this out) ----------

# N_households = 20_000
# agents = initialize_agents(tracts, N_households, seed=123)
# agents_final, omega_path, history = run_simulation(tracts, agents, T=20, seed=42)
