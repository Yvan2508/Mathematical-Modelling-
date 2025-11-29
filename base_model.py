"""
Simplified baseline model where agents move based ONLY on own-group share.

Agents ignore rents, distance, amenities, and income dynamics. Each period,
an agent compares the share of their own group in every tract with available
capacity and moves to a tract that maximizes that share (if it is strictly
higher than the share in their current tract) and they are dissatisfied with
their current own-group share. Capacity constraints are respected and shares
are updated online as agents move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynamics import compute_tract_occupancy

TOLERANCE_LEVELS = {
    "very_high": 0.25,  # very tolerant of mixed tracts
    "medium": 0.50,
    "very_low": 0.75,   # wants high own-group share
}


def one_period_update_base(
    agents: pd.DataFrame,
    tracts,
    omega_t: np.ndarray | None,
    rng: np.random.Generator,
    tolerance: float | str = "medium",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-period update for the simplified model:
      - Agents move only to increase own-group share.
      - Ignores rents, distance, income mobility, and other utilities.
      - Respects tract capacity.

    Parameters
    ----------
    tolerance : float or str
        Minimum own-group share an agent wants. Use "very_high" (0.25),
        "medium" (0.50), or "very_low" (0.75), or pass a custom float in [0,1].
    """
    N_tr = len(tracts)
    if omega_t is None:
        omega_t = np.zeros(N_tr, dtype=float)

    K = tracts["K"].values.astype(float)

    if isinstance(tolerance, str):
        if tolerance not in TOLERANCE_LEVELS:
            raise ValueError(f"Unknown tolerance '{tolerance}'. Valid: {list(TOLERANCE_LEVELS.keys())}")
        tol_value = TOLERANCE_LEVELS[tolerance]
    else:
        tol_value = float(tolerance)
    tol_value = float(np.clip(tol_value, 0.0, 1.0))

    # Occupancy and shares at start of period
    H_n, H_ng, s_ng = compute_tract_occupancy(agents, tracts)

    agent_indices = np.arange(len(agents))
    rng.shuffle(agent_indices)

    def _recompute_shares_for(tract_idx: int):
        """Update shares for a single tract after a move."""
        if H_n[tract_idx] > 0:
            s_ng[tract_idx, :] = H_ng[tract_idx, :] / H_n[tract_idx]
        else:
            s_ng[tract_idx, :] = 0.0

    for idx in agent_indices:
        g_idx = int(agents.at[idx, "group_idx"])
        current_t = int(agents.at[idx, "tract_idx"])

        has_space = H_n < K
        # Staying put is always allowed even if tract is currently full
        has_space[current_t] = True

        shares = np.nan_to_num(s_ng[:, g_idx], nan=0.0)

        feasible_shares = shares.copy()
        feasible_shares[~has_space] = -np.inf

        best_share = feasible_shares.max()
        if not np.isfinite(best_share):
            continue  # nowhere to go

        current_share = shares[current_t]
        if current_share >= tol_value:
            continue  # already satisfied
        if best_share <= current_share:
            continue  # no improvement available

        # Prefer candidates meeting tolerance; fallback to best improvement
        meets_tol = feasible_shares >= tol_value
        if meets_tol.any():
            candidate_pool = feasible_shares.copy()
            candidate_pool[~meets_tol] = -np.inf
            target_max = candidate_pool.max()
            best_candidates = np.where(np.isclose(candidate_pool, target_max))[0]
        else:
            best_candidates = np.where(np.isclose(feasible_shares, best_share))[0]

        new_t = int(rng.choice(best_candidates))

        if new_t == current_t:
            continue

        # Update occupancy and counts
        H_n[current_t] -= 1
        H_ng[current_t, g_idx] -= 1

        H_n[new_t] += 1
        H_ng[new_t, g_idx] += 1

        agents.at[idx, "tract_idx"] = new_t
        agents.at[idx, "tract_id"] = tracts["tract_id"].iloc[new_t]

        # Refresh shares for the two affected tracts
        _recompute_shares_for(current_t)
        _recompute_shares_for(new_t)

    omega_next = omega_t.copy()
    return agents, omega_next, H_n, H_ng, s_ng


def run_base_simulation(
    tracts,
    agents: pd.DataFrame,
    T: int = 20,
    seed: int = 0,
    tolerance: float | str = "medium",
) -> tuple[pd.DataFrame, list[np.ndarray], list[dict]]:
    """
    Run the base model for T periods.

    Returns
    -------
    agents_final : DataFrame
    omega_path   : list of rent vectors (length T+1, but rents stay fixed)
    history      : list of dict snapshots with H_n, H_ng, s_ng each period
    """
    rng = np.random.default_rng(seed)

    omega_t = np.zeros(len(tracts), dtype=float)
    omega_path = [omega_t.copy()]

    H_n0, H_ng0, s_ng0 = compute_tract_occupancy(agents, tracts)
    history = [{"H_n": H_n0, "H_ng": H_ng0, "s_ng": s_ng0}]

    for t in range(T):
        print(f"[Base model] Period {t+1}/{T}")
        agents, omega_t, H_n, H_ng, s_ng = one_period_update_base(
            agents, tracts, omega_t, rng, tolerance=tolerance
        )
        omega_path.append(omega_t.copy())
        history.append({"H_n": H_n, "H_ng": H_ng, "s_ng": s_ng})

    return agents, omega_path, history


if __name__ == "__main__":
    """
    Run the simplified base model with its own defaults and emit animations.
    Only own-group share matters; rents/amenities/income are ignored.
    """
    from pathlib import Path

    tracts_path = Path("data") / "chicago_tracts_enriched.shp"
    try:
        import geopandas as gpd
        from agents import initialize_agents
        from animate_agents import (
            run_simulation_with_agent_history,
            create_animated_html_map,
            create_animated_video,
        )
    except ImportError:
        raise SystemExit("Install geopandas, shapely, matplotlib, and folium to run the base model animation.")

    # Base-model defaults (adjust as needed)
    tolerance = "very_high"      # own-group share threshold ("very_high", "medium", "very_low", or float)
    T_periods = 40            # number of periods to simulate
    rng_seed = 1              # seed for relocation randomness
    agent_seed = 42           # seed for initial agent placement
    N_households = 10_000     # cap agent count to keep base model runtime similar to full model
    occupancy_rate = 0.95     # ignored when N_households is provided

    if not tracts_path.exists():
        raise SystemExit(f"Tracts file not found at {tracts_path}. Run prepare_tract.py first.")

    print("Loading tracts for base model...")
    tracts = gpd.read_file(tracts_path)

    print(f"Initializing agents (N_households={N_households}, occupancy_rate={occupancy_rate})...")
    agents_init = initialize_agents(
        tracts,
        N_households=N_households,
        occupancy_rate=occupancy_rate,
        seed=agent_seed
    )
    print(f"Initialized {len(agents_init)} agents across {len(tracts)} tracts.")

    print(f"Running base model for {T_periods} periods with tolerance='{tolerance}'...")
    agent_history = run_simulation_with_agent_history(
        agents_init,
        tracts,
        T=T_periods,
        seed=rng_seed,
        use_base_model=True,
        tolerance=tolerance,
    )
    agents_final = agent_history[-1]
    print(f"Completed {T_periods} periods with {len(agents_final)} agents.")

    # Create animation outputs
    output_dir = Path("data") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"base_model_animation_T{T_periods}_tol_{tolerance}.html"
    print(f"Creating HTML animation at {html_path}...")
    create_animated_html_map(agent_history, tracts, html_path, sample_rate=1)

    video_path = output_dir / f"base_model_animation_T{T_periods}_tol_{tolerance}.gif"
    try:
        print(f"Creating GIF animation at {video_path}...")
        create_animated_video(agent_history, tracts, video_path, sample_rate=1, fps=5)
    except Exception as e:
        print(f"Could not create GIF animation: {e}")
        print("HTML animation created successfully.")
