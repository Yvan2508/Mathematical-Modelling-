# Chicago Census Tract Data Preparation for Segregation ABM Model

## Overview

This project prepares Chicago census tract data with initial conditions and attributes for an Agent-Based Model (ABM) of residential segregation. The script loads census tract boundaries, calculates spatial attributes, fetches real demographic and housing data from the U.S. Census Bureau, and generates comprehensive quality-of-life indicators.

## Purpose

The `prepare_tract.py` script processes Chicago census tracts to create a dataset with:
- Spatial attributes (commuting distance, park access, transit access)
- Housing characteristics (median rent, household capacity)
- Demographic initial conditions (initial group shares)
- Comprehensive amenity index combining multiple quality-of-life factors

## Data Sources

### Geographic Data
- **Census Tract Boundaries**: TIGER/Line shapefiles from U.S. Census Bureau (2020)
- **Location**: Cook County, Illinois (State FIPS: 17, County FIPS: 031)
- **Filtering**: Tracts are filtered to include only those within Chicago city boundaries

### Demographic and Housing Data
- **Source**: American Community Survey (ACS) 5-year estimates, 2019
- **API**: U.S. Census Bureau API (no API key required for basic usage)
- **Variables Fetched**:
  - Housing: Median rent (B25064_001E), housing units, owner/renter occupancy
  - Demographics: Race/ethnicity (White, Black, Asian)
  - Education: Educational attainment (proxy for school quality)
  - Economic: Median household income, public transit usage

## Key Variables and Calculations

### 1. Commuting Distance (`dw`)
- **Definition**: Distance from tract centroid to 3 main commerce areas in Chicago (The Loop, O'Hare, Elk Grove, Calumet)
- **Location**: Point(-87.6298, 41.8781), Point(-87.9090, 41.9810), Point(-87.9700, 41.9900), Point(-87.5700, 41.6900) 
- **Method**: Calculated in UTM Zone 16N (EPSG:26916) for accurate metric distances
- **Units**: Kilometers

### 2. Median Rent (`p0`)
- **Source**: ACS variable B25064_001E (Median gross rent)
- **Units**: USD per month
- **Handling**: Census special values (-666666666, -999999999, -222222222) replaced with median

### 3. Household Capacity (`K`)
- **Source**: ACS variable B25001_001E (Total housing units)
- **Definition**: Maximum number of households that can live in the tract (occupied + vacant units)
- **Handling**: Missing values filled with median. Tracts with K=0 are filtered out (non-residential areas)

### 4. Initial Group Shares (`s_B0`, `s_W0`, `s_A0`, `s_H0`)
- **Definition**: Initial proportion of households in each demographic group
- **IMPORTANT**: Groups are MUTUALLY EXCLUSIVE - each household belongs to exactly ONE group
- **Groups**: 
  - `s_B0`: Non-Hispanic Black or African American
  - `s_W0`: Non-Hispanic White
  - `s_A0`: Non-Hispanic Asian
  - `s_H0`: Hispanic or Latino (of any race) - takes priority
- **Source**: ACS variables B02001_002E, B02001_003E, B02001_005E, B03003_003E
- **Method**: Hierarchical assignment - Hispanic ethnicity takes priority, then non-Hispanic individuals are allocated to race groups (B, W, A) based on race proportions
- **Normalization**: Shares sum to 1.0 for each tract (ensures mutually exclusive groups)

### 5. Comprehensive Amenity Index (`amenity` or `v_n`)

The amenity index combines seven quality-of-life factors with the following weights:

| Factor | Weight | Calculation Method |
|--------|--------|-------------------|
| **School Quality** | 20% | Education attainment rate (% with bachelor's degree or higher) |
| **Economic Well-being** | 20% | Median household income (normalized) |
| **Public Transit Access** | 15% | 60% transit usage + 40% proximity to CTA L stations |
| **Median Rent** | 15% | Normalized median gross rent |
| **Owner-Occupancy** | 10% | Ratio of owner-occupied to total occupied units |
| **Parks & Recreation** | 10% | Inverse distance to major Chicago parks |
| **Crime Rate** | 10% | Inverse of income (placeholder; replace with actual crime data) |

**Parks Included**:
- Millennium Park, Grant Park, Lincoln Park, Jackson Park, Washington Park

**Transit Hubs Included**:
- State/Lake, Washington/Wabash, Clark/Lake (Loop stations)
- Fullerton, Belmont, Diversey (North Side stations)

**Normalization**: All factors normalized to [0, 1] range, then combined and scaled to **[-0.5, 0.5]**

## How to Run

### Prerequisites

```bash
pip install numpy pandas geopandas shapely matplotlib folium requests
```

### Required Data Files

The script expects a shapefile at `data/chicago_tracts.shp` containing Chicago census tracts. This should be a pre-filtered shapefile containing only Chicago tracts (not all Cook County tracts).

### Execution

```bash
python prepare_tract.py
```

### Choosing model dynamics

- **Full dynamics (default):** Uses rents, distance, income mobility, and social preferences (`dynamics.py`).
- **Base model (share-only):** Agents move only to increase own-group share, ignoring rents/distance/mobility (`base_model.py`).

Switch in `animate_agents.py` by setting `use_base_model`. Example:

```python
agent_history = run_simulation_with_agent_history(
    agents,
    tracts,
    T=20,
    seed=123,
    use_base_model=True,  # False for full dynamics
)
```

For the base model you can set an own-group share tolerance via `tolerance`:
`"very_high"` = 0.25, `"medium"` = 0.50, `"very_low"` = 0.75 (or pass a custom float).

### Configuration

In the main execution block, set:
- `USE_REAL_DATA = True`: Fetch real data from Census API
- `USE_REAL_DATA = False`: Use placeholder/synthetic data

## Output Files

### Data Files
- **`data/chicago_tracts_enriched.shp`**: Shapefile with all calculated attributes
  - Columns: `tract_id`, `geometry`, `dw`, `p0`, `K`, `amenity`, `s_B0`, `s_W0`, `s_A0`

### Visualization Files (in `data/figures/`)
- **Static Maps** (PNG):
  - `chicago_commute_distance.png`: Commuting distance to downtown
  - `chicago_baseline_rent.png`: Median rent distribution
  - `chicago_capacity.png`: Household capacity
  - `chicago_amenity_index.png`: Comprehensive amenity index
  - `chicago_group_B_share.png`: Initial share of Group B
  - `chicago_group_W_share.png`: Initial share of Group W
  - `chicago_group_A_share.png`: Initial share of Group A

- **Interactive Map** (HTML):
  - `chicago_tracts_interactive.html`: Folium map with toggleable layers for all attributes
  - Features: Multiple base maps, tooltips with attribute values, layer controls

## Key Design Decisions

1. **Census API vs. Placeholder Data**: The script supports both real Census data and placeholder data for testing. Real data uses 2019 ACS 5-year estimates.

2. **Amenity Index Weights**: Weights were chosen to balance economic factors (income, rent) with quality-of-life factors (schools, transit, parks). School quality and economic well-being receive the highest weights (20% each).

3. **Transit Access Calculation**: Combines both usage patterns (from Census) and spatial proximity (distance to stations) for a more comprehensive measure.

4. **Crime Rate Proxy**: Currently uses inverse of income as a placeholder. Should be replaced with actual crime data when available.

5. **Lake Michigan Exclusion**: Visualizations exclude Lake Michigan by setting plot bounds to city boundary only.

6. **Data Cleaning**: Census special values (indicating suppressed or unavailable data) are replaced with medians to maintain tract-level analysis.

## Reproducibility

- **Census Data Year**: 2019 ACS 5-year estimates
- **Shapefile Year**: 2020 TIGER/Line
- **Random Seeds**: Placeholder data uses seed=0 for reproducibility
- **Coordinate System**: WGS84 (EPSG:4326) for geographic data, UTM Zone 16N (EPSG:26916) for distance calculations

## Notes

- The script handles missing data by filling with medians after removing Census special values
- Group shares are normalized to ensure they sum to 1.0 for each tract
- All distance calculations use metric CRS (UTM) for accuracy
- The interactive map restricts panning to Chicago city bounds to exclude Lake Michigan
