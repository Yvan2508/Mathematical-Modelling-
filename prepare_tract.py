"""
STEP 1: Define tracts + attributes for the segregation ABM model
----------------------------------------------------------------
This script:
  - Loads Chicago census tracts (Shapefile)
  - Standardizes tract ID
  - Computes commuting distance to downtown (dw)
  - Fetches real census/ACS data OR uses placeholder values for:
    * p0: Median rent (from ACS B25064_001E)
    * K: Household capacity (from ACS B25001_001E - total housing units)
    * amenity: Amenity index (derived from rent and owner-occupancy)
    * s_B0, s_W0, s_A0, s_H0: Initial group shares (from ACS race/ethnicity data)

To use real data, set USE_REAL_DATA = True in the main execution block.
The code uses the Census API directly (no API key required for basic usage).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


# ================================
# Load Chicago census tracts
# ================================

def load_chicago_tracts(shp_path=None):
    """
    Load Chicago census tract shapefile and standardize tract ID
    
    Parameters:
    -----------
    shp_path : str or Path, optional
        Path to the shapefile. If None, uses default location: data/chicago_tracts.shp
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame containing Chicago census tracts with standardized 'tract_id' column
    """
    if shp_path is None:
        shp_path = Path("data") / "chicago_tracts.shp"
    else:
        shp_path = Path(shp_path)
    
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found at {shp_path}. "
            "Please ensure the shapefile has been downloaded and saved to this location."
        )
    
    print(f"Loading Chicago census tracts from {shp_path}...")
    gdf = gpd.read_file(shp_path)
    print(f"Loaded {len(gdf)} census tracts")
    print(f"CRS: {gdf.crs}")

    # Standardize tract_id column name
    if "tract_id" in gdf.columns:
        tracts = gdf[["tract_id", "geometry"]].copy()
    elif "GEOID" in gdf.columns:
        tracts = gdf[["GEOID", "geometry"]].rename(columns={"GEOID": "tract_id"})
    else:
        raise ValueError("Input file must contain column 'tract_id' or 'GEOID'.")

    tracts = tracts.reset_index(drop=True)
    return tracts


# ================================
# Get city boundary for clipping
# ================================

def get_city_boundary(tracts: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Get Chicago city boundary from the union of all tracts.
    This creates a boundary that excludes water areas like Lake Michigan.
    """
    # Create a union of all tract geometries to form the city boundary
    city_boundary = tracts.geometry.unary_union
    
    # Convert to GeoSeries for easier use
    boundary_gdf = gpd.GeoSeries([city_boundary], crs=tracts.crs)
    
    return boundary_gdf


# ================================
# Add commuting cost proxy
# ================================

def add_commute_cost(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add commuting cost based on centroid distance to multiple Chicago employment centers (km).
    Calculates distance to 4 major employment centroids and uses the minimum distance.
    """
    tracts = tracts.copy()

    # Ensure coordinate reference system exists
    if tracts.crs is None:
        tracts.set_crs(epsg=4326, inplace=True)

    # Define multiple employment centroids
    employment_centers = {
        'Loop': Point(-87.6298, 41.8781),           # Chicago downtown (The Loop)
        'OHare': Point(-87.9090, 41.9810),          # O'Hare area
        'Elk_Grove': Point(-87.9700, 41.9900),      # Elk Grove
        'Calumet': Point(-87.5700, 41.6900)         # Calumet
    }
    
    # Convert employment centers to GeoDataFrame
    centers_gdf = gpd.GeoDataFrame(
        geometry=list(employment_centers.values()),
        index=list(employment_centers.keys()),
        crs="EPSG:4326"
    )

    # Convert to metric CRS for distance calculations
    metric_crs = 26916  # UTM Zone 16N (covers Chicago region)
    tracts_metric = tracts.to_crs(epsg=metric_crs)
    centers_metric = centers_gdf.to_crs(epsg=metric_crs)

    # Compute centroid distances in km
    centroids = tracts_metric.geometry.centroid
    
    # Calculate distance to each employment center
    distances = {}
    for center_name, center_geom in centers_metric.geometry.items():
        dist_meters = centroids.distance(center_geom)
        distances[f'dist_{center_name}'] = dist_meters / 1000.0
    
    # Add individual distances to tracts (optional, for analysis)
    for col_name, dist_values in distances.items():
        tracts[col_name] = dist_values
    
    # Calculate minimum distance to any employment center
    dist_df = pd.DataFrame(distances)
    tracts["dw"] = dist_df.min(axis=1)
    
    # Optionally, record which center is closest
    tracts["closest_center"] = dist_df.idxmin(axis=1).str.replace('dist_', '')

    return tracts


# ================================
# Calculate transit access
# ================================

def add_transit_access(tracts: gpd.GeoDataFrame, transit_points: list) -> gpd.GeoDataFrame:
    """
    Calculate transit access as minimum distance to major transit stops.
    
    Parameters:
    -----------
    tracts : gpd.GeoDataFrame
        Tracts with geometry
    transit_points : list
        List of Point geometries for major transit stops
    
    Returns:
    --------
    gpd.GeoDataFrame
        Tracts with 'transit_distance' column (distance in km)
    """
    tracts = tracts.copy()
    
    # Ensure tracts are in a metric CRS for distance calculation
    if tracts.crs != 26916:  # UTM Zone 16N
        tracts_metric = tracts.to_crs(epsg=26916)
    else:
        tracts_metric = tracts
    
    # Convert transit points to metric CRS
    transit_gdf = gpd.GeoDataFrame(geometry=transit_points, crs="EPSG:4326")
    transit_metric = transit_gdf.to_crs(epsg=26916)
    
    # Calculate minimum distance from each tract centroid to any transit stop
    centroids = tracts_metric.geometry.centroid
    min_distances = []
    
    for centroid in centroids:
        distances = [centroid.distance(stop) for stop in transit_metric.geometry]
        min_distances.append(min(distances) / 1000.0)  # Convert to km
    
    tracts['transit_distance'] = min_distances
    
    return tracts


# ================================
# Calculate park access
# ================================

def add_park_access(tracts: gpd.GeoDataFrame, park_points: list) -> gpd.GeoDataFrame:
    """
    Calculate park access as minimum distance to major parks.
    
    Parameters:
    -----------
    tracts : gpd.GeoDataFrame
        Tracts with geometry
    park_points : list
        List of Point geometries for major parks
    
    Returns:
    --------
    gpd.GeoDataFrame
        Tracts with 'park_access' column (distance in km)
    """
    tracts = tracts.copy()
    
    # Ensure tracts are in a metric CRS for distance calculation
    if tracts.crs != 26916:  # UTM Zone 16N
        tracts_metric = tracts.to_crs(epsg=26916)
    else:
        tracts_metric = tracts
    
    # Convert park points to metric CRS
    parks_gdf = gpd.GeoDataFrame(geometry=park_points, crs="EPSG:4326")
    parks_metric = parks_gdf.to_crs(epsg=26916)
    
    # Calculate minimum distance from each tract centroid to any park
    centroids = tracts_metric.geometry.centroid
    min_distances = []
    
    for centroid in centroids:
        distances = [centroid.distance(park) for park in parks_metric.geometry]
        min_distances.append(min(distances) / 1000.0)  # Convert to km
    
    tracts['park_access'] = min_distances
    
    return tracts


# ================================
# Fetch real census/ACS data
# ================================

def fetch_real_census_data(tracts: gpd.GeoDataFrame, 
                           use_real_data: bool = True) -> gpd.GeoDataFrame:
    """
    Fetch real census/ACS data for tracts.
    
    Parameters:
    -----------
    tracts : gpd.GeoDataFrame
        Tracts with tract_id column (GEOID format)
    use_real_data : bool
        If True, fetch real data from Census API. If False, use placeholders.
    
    Returns:
    --------
    gpd.GeoDataFrame
        Tracts with real or placeholder attributes
    """
    tracts = tracts.copy()
    
    if not use_real_data:
        # Use placeholder data
        return add_placeholder_attributes(tracts)
    
    try:
        import requests
        
        print("Fetching real census data from Census API...")
        
        # Variables to fetch from ACS 5-year estimates:
        # Housing and economic:
        # B25064_001E: Median gross rent
        # B25001_001E: Total housing units
        # B25003_002E: Owner-occupied housing units
        # B25003_003E: Renter-occupied housing units
        # B25003_001E: Total occupied housing units
        # B19013_001E: Median household income
        # B08301_021E: Public transportation to work
        # B08301_001E: Total workers
        
        # Demographics:
        # B02001_002E: White alone
        # B02001_003E: Black or African American alone
        # B02001_005E: Asian alone
        # B03003_003E: Hispanic or Latino (of any race)
        # B03003_001E: Total population for Hispanic ethnicity calculation
        
        # Education (proxy for school quality):
        # B15003_022E: Bachelor's degree
        # B15003_023E: Master's degree
        # B15003_024E: Professional degree
        # B15003_025E: Doctorate degree
        # B15003_001E: Total population 25+ with education
        
        variables = {
            'median_rent': 'B25064_001E',
            'total_units': 'B25001_001E',
            'owner_occupied': 'B25003_002E',
            'renter_occupied': 'B25003_003E',
            'white': 'B02001_002E',
            'black': 'B02001_003E',
            'asian': 'B02001_005E',
            'hispanic': 'B03003_003E',
            'total_pop_ethnicity': 'B03003_001E',
            'total_occupied': 'B25003_001E',
            'median_income': 'B19013_001E',
            'public_transit_workers': 'B08301_021E',
            'total_workers': 'B08301_001E',
            'bachelors': 'B15003_022E',
            'masters': 'B15003_023E',
            'professional': 'B15003_024E',
            'doctorate': 'B15003_025E',
            'total_educated': 'B15003_001E'
        }
        
        # Census API endpoint for ACS 5-year estimates
        # Using 2019 ACS 5-year estimates (most recent available via API)
        base_url = "https://api.census.gov/data/2019/acs/acs5"
        
        # Get all variable names
        var_list = ','.join(['NAME', 'GEO_ID'] + list(variables.values()))
        
        # Cook County, IL: state=17, county=031
        # Get all tracts in Cook County
        params = {
            'get': var_list,
            'for': 'tract:*',
            'in': 'state:17 county:031'
        }
        
        print("Downloading ACS 2019 5-year estimates for Cook County tracts...")
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"Census API returned status code {response.status_code}")
        
        # Parse JSON response
        data_json = response.json()
        
        # Convert to DataFrame
        headers = data_json[0]
        data_rows = data_json[1:]
        data = pd.DataFrame(data_rows, columns=headers)
        
        # Extract GEOID from GEO_ID (format: "1400000US17031010100")
        data['GEOID'] = data['GEO_ID'].str.replace('1400000US', '')
        
        # Convert variable columns to numeric
        for var in variables.values():
            data[var] = pd.to_numeric(data[var], errors='coerce')
        
        # Replace Census API special values with NaN
        # -666666666: Not available, -999999999: Suppressed, -222222222: Not available
        census_na_values = [-666666666, -999999999, -222222222]
        for var in variables.values():
            data[var] = data[var].replace(census_na_values, np.nan)
        
        # Merge with tracts using GEOID
        tracts['GEOID'] = tracts['tract_id'].astype(str)
        data['GEOID'] = data['GEOID'].astype(str)
        
        # Merge
        tracts = tracts.merge(data[['GEOID'] + list(variables.values())], 
                             on='GEOID', how='left')
        
        # Map to model variables
        # p0: Median rent (convert to monthly if needed, already in dollars)
        tracts['omega_bar'] = pd.to_numeric(tracts[variables['median_rent']], errors='coerce')
        
        # Replace any remaining Census special values in merged data
        tracts['omega_bar'] = tracts['omega_bar'].replace(census_na_values, np.nan)
        
        # K: Household capacity (use total housing units = occupied + vacant)
        # This represents the maximum number of households that can live in the tract
        tracts['K'] = pd.to_numeric(tracts[variables['total_units']], errors='coerce')
        tracts['K'] = tracts['K'].replace(census_na_values, np.nan)
        
        # Group shares: Calculate proportions for 4 MUTUALLY EXCLUSIVE groups (B, W, A, H)
        # IMPORTANT: Each household belongs to exactly ONE group.
        # Strategy: Hispanic (H) takes priority, then non-Hispanic race categories (B, W, A)
        white = pd.to_numeric(tracts[variables['white']], errors='coerce')
        white = white.replace(census_na_values, np.nan).fillna(0)
        black = pd.to_numeric(tracts[variables['black']], errors='coerce')
        black = black.replace(census_na_values, np.nan).fillna(0)
        asian = pd.to_numeric(tracts[variables['asian']], errors='coerce')
        asian = asian.replace(census_na_values, np.nan).fillna(0)
        hispanic = pd.to_numeric(tracts[variables['hispanic']], errors='coerce')
        hispanic = hispanic.replace(census_na_values, np.nan).fillna(0)
        
        # Use total population from ethnicity table for accurate denominator
        total_pop_eth = pd.to_numeric(tracts[variables['total_pop_ethnicity']], errors='coerce')
        total_pop_eth = total_pop_eth.replace(census_na_values, np.nan)
        
        # Fallback to sum of categories if ethnicity total is missing
        total_pop = white + black + asian + hispanic
        total_pop = np.where(total_pop_eth > 0, total_pop_eth, total_pop)
        
        # Calculate mutually exclusive group shares using hierarchical approach:
        # 1. Hispanic (H) takes priority - all Hispanic individuals go to group H
        # 2. Non-Hispanic individuals are allocated to B, W, A based on race
        
        # Non-Hispanic population
        non_hispanic_pop = total_pop - hispanic
        
        # For non-Hispanic groups, we need to estimate non-Hispanic counts for each race
        # Since race categories (white, black, asian) include Hispanic individuals,
        # we approximate non-Hispanic counts by assuming race distribution among
        # non-Hispanics is proportional to overall race distribution, scaled to non-Hispanic pop
        
        race_total = white + black + asian
        
        # Calculate shares: Hispanic first, then allocate non-Hispanic by race proportions
        tracts['s_H0'] = np.where(total_pop > 0, hispanic / total_pop, 0.25)
        
        # For non-Hispanic groups, use race proportions scaled to non-Hispanic population
        # This ensures: s_H0 + s_W0 + s_B0 + s_A0 = 1 (mutually exclusive)
        if np.any(race_total > 0):
            # Proportion of each race among total race population
            white_prop = np.where(race_total > 0, white / race_total, 1/3)
            black_prop = np.where(race_total > 0, black / race_total, 1/3)
            asian_prop = np.where(race_total > 0, asian / race_total, 1/3)
            
            # Allocate non-Hispanic population proportionally to race distribution
            # This gives us non-Hispanic counts for each race group
            non_hispanic_white_est = non_hispanic_pop * white_prop
            non_hispanic_black_est = non_hispanic_pop * black_prop
            non_hispanic_asian_est = non_hispanic_pop * asian_prop
            
            # Calculate shares (as fraction of total population)
            tracts['s_W0'] = np.where(total_pop > 0, non_hispanic_white_est / total_pop, 0.25)
            tracts['s_B0'] = np.where(total_pop > 0, non_hispanic_black_est / total_pop, 0.25)
            tracts['s_A0'] = np.where(total_pop > 0, non_hispanic_asian_est / total_pop, 0.25)
        else:
            # Fallback if no race data
            tracts['s_W0'] = np.where(total_pop > 0, non_hispanic_pop / (3 * total_pop), 0.25)
            tracts['s_B0'] = np.where(total_pop > 0, non_hispanic_pop / (3 * total_pop), 0.25)
            tracts['s_A0'] = np.where(total_pop > 0, non_hispanic_pop / (3 * total_pop), 0.25)
        
        # Normalize to ensure they sum to exactly 1 (mutually exclusive groups)
        total_share = tracts[['s_B0', 's_W0', 's_A0', 's_H0']].sum(axis=1)
        tracts['s_B0'] = tracts['s_B0'] / total_share
        tracts['s_W0'] = tracts['s_W0'] / total_share
        tracts['s_A0'] = tracts['s_A0'] / total_share
        tracts['s_H0'] = tracts['s_H0'] / total_share
        
        # Calculate additional quality-of-life indicators
        
        # 1. School Quality (proxy: education attainment rate)
        bachelors = pd.to_numeric(tracts[variables['bachelors']], errors='coerce').replace(census_na_values, np.nan)
        masters = pd.to_numeric(tracts[variables['masters']], errors='coerce').replace(census_na_values, np.nan)
        professional = pd.to_numeric(tracts[variables['professional']], errors='coerce').replace(census_na_values, np.nan)
        doctorate = pd.to_numeric(tracts[variables['doctorate']], errors='coerce').replace(census_na_values, np.nan)
        total_educated = pd.to_numeric(tracts[variables['total_educated']], errors='coerce').replace(census_na_values, np.nan)
        
        college_plus = bachelors + masters + professional + doctorate
        school_quality = np.where(total_educated > 0, college_plus / total_educated, 0.3)
        school_quality = pd.Series(school_quality, index=tracts.index)
        
        # 2. Public Transit Access 
        # Use both: % of workers using public transit AND distance to transit stops
        transit_workers = pd.to_numeric(tracts[variables['public_transit_workers']], errors='coerce').replace(census_na_values, np.nan)
        total_workers = pd.to_numeric(tracts[variables['total_workers']], errors='coerce').replace(census_na_values, np.nan)
        transit_usage = np.where(total_workers > 0, transit_workers / total_workers, 0.1)
        
        # Also calculate distance to major transit hubs (CTA L stations)
        major_transit_hubs = [
            Point(-87.6248, 41.8789),  # State/Lake (Loop)
            Point(-87.6327, 41.8789),  # Washington/Wabash (Loop)
            Point(-87.6277, 41.8789),  # Clark/Lake (Loop)
            Point(-87.6534, 41.8815),  # Fullerton (Red/Brown)
            Point(-87.6681, 41.9097),  # Belmont (Red/Brown)
            Point(-87.6552, 41.9037),  # Diversey (Brown)
        ]
        tracts = add_transit_access(tracts, major_transit_hubs)
        
        # Combine transit usage and proximity (inverse distance = better)
        transit_proximity = 1.0 / (tracts['transit_distance'] + 0.1)  # Add small value to avoid division by zero
        transit_proximity_norm = (transit_proximity - transit_proximity.min()) / \
                                (transit_proximity.max() - transit_proximity.min() + 1e-10)
        transit_access = (transit_usage * 0.6 + transit_proximity_norm * 0.4)  # Weighted combination
        transit_access = pd.Series(transit_access, index=tracts.index)
        
        # 3. Economic Well-being (median household income)
        median_income = pd.to_numeric(tracts[variables['median_income']], errors='coerce').replace(census_na_values, np.nan)
        
        # 4. Parks and Recreation (calculate distance to major parks)
        # Chicago's major parks coordinates
        major_parks = [
            Point(-87.6244, 41.8781),  # Millennium Park
            Point(-87.6334, 41.8781),  # Grant Park
            Point(-87.6534, 41.8815),  # Lincoln Park
            Point(-87.5967, 41.7897),  # Jackson Park
            Point(-87.5847, 41.7897),  # Washington Park
        ]
        tracts = add_park_access(tracts, major_parks)
        
        # 5. Crime rates (placeholder - would need actual crime data)
        # For now, use inverse of income as proxy (higher income = lower crime generally)
        crime_rate = 1.0 - (median_income.fillna(median_income.median()) / median_income.fillna(median_income.median()).max())
        crime_rate = crime_rate.fillna(0.5)
        
        # Store intermediate variables for amenity calculation
        owner_occ = pd.to_numeric(tracts[variables['owner_occupied']], errors='coerce')
        owner_occ = owner_occ.replace(census_na_values, np.nan)
        total_occ = pd.to_numeric(tracts[variables['total_occupied']], errors='coerce')
        total_occ = total_occ.replace(census_na_values, np.nan)
        owner_ratio = owner_occ / total_occ
        owner_ratio = owner_ratio.fillna(0.5)
        
        # Fill missing values with median before normalization
        tracts['omega_bar'] = tracts['omega_bar'].fillna(tracts['omega_bar'].median())
        tracts['K'] = tracts['K'].fillna(tracts['K'].median())
        median_income = median_income.fillna(median_income.median())
        school_quality = school_quality.fillna(school_quality.median())
        transit_access = transit_access.fillna(transit_access.median())
        crime_rate = crime_rate.fillna(0.5)
        
        # Normalize all factors to [0, 1] range
        def normalize_series(series, inverse=False):
            """Normalize a series to [0, 1], optionally inverting (for distance/crime)."""
            valid = series.dropna()
            if len(valid) == 0:
                return pd.Series(0.5, index=series.index)
            min_val, max_val = valid.min(), valid.max()
            if max_val - min_val < 1e-10:
                return pd.Series(0.5, index=series.index)
            norm = (series - min_val) / (max_val - min_val)
            if inverse:
                norm = 1.0 - norm  # Invert so lower values = better
            return norm.fillna(0.5)
        
        # Normalize each factor
        rent_norm = normalize_series(tracts['omega_bar'])
        owner_norm = normalize_series(owner_ratio)
        income_norm = normalize_series(median_income)
        school_norm = normalize_series(school_quality)
        transit_norm = normalize_series(transit_access)
        park_norm = normalize_series(tracts['park_access'], inverse=True)  # Closer = better
        crime_norm = normalize_series(crime_rate, inverse=True)  # Lower crime = better
        
        # Combine into amenity index with weights
        # Weights sum to 1.0, final index scaled to [-0.5, 0.5]
        weights = {
            'rent': 0.15,           # Median rent
            'owner': 0.10,          # Owner-occupancy
            'income': 0.20,         # Economic well-being
            'school': 0.20,         # School quality (education)
            'transit': 0.15,        # Public transit access
            'park': 0.10,           # Park access
            'crime': 0.10           # Crime rate (inverse)
        }
        
        tracts['v_n'] = (
            rent_norm * weights['rent'] +
            owner_norm * weights['owner'] +
            income_norm * weights['income'] +
            school_norm * weights['school'] +
            transit_norm * weights['transit'] +
            park_norm * weights['park'] +
            crime_norm * weights['crime']
        ) - 0.5  # Scale to [-0.5, 0.5]
        
        tracts['v_n'] = tracts['v_n'].fillna(0.0)
        
        print(f"Successfully fetched real data for {len(tracts)} tracts")
        print(f"Median rent range: ${tracts['omega_bar'].min():.0f} - ${tracts['omega_bar'].max():.0f}")
        print(f"Capacity range: {tracts['K'].min():.0f} - {tracts['K'].max():.0f}")
        
        return tracts
        
    except ImportError:
        print("Note: requests library not available. Install with: pip install requests")
        print("Using placeholder data instead.")
        tracts = add_placeholder_attributes(tracts)
        tracts = add_initial_group_shares(tracts)
        return tracts
    except Exception as e:
        print(f"Error fetching real census data: {e}")
        print("Using placeholder data instead.")
        tracts = add_placeholder_attributes(tracts)
        tracts = add_initial_group_shares(tracts)
    return tracts


# ================================
# Calculate comprehensive amenity index
# ================================

def calculate_comprehensive_amenity_index(tracts: gpd.GeoDataFrame,
                                         use_real_factors: bool = False) -> gpd.GeoDataFrame:
    """
    Calculate comprehensive amenity index from multiple quality-of-life factors.
    
    Parameters:
    -----------
    tracts : gpd.GeoDataFrame
        Tracts with required attributes
    use_real_factors : bool
        If True, uses real factors. If False, creates placeholder factors.
    
    Returns:
    --------
    gpd.GeoDataFrame
        Tracts with 'v_n' (amenity index) column
    """
    tracts = tracts.copy()
    
    if use_real_factors:
        # Use real factors that should already be in tracts
        # Factors: rent, owner_ratio, income, school_quality, transit_access, park_access, crime_rate
        # These should be calculated in fetch_real_census_data
        pass  # Already calculated in fetch_real_census_data
    else:
        # For placeholder data, add park access and create simple amenity index
        major_parks = [
            Point(-87.6244, 41.8781),  # Millennium Park
            Point(-87.6334, 41.8781),  # Grant Park
            Point(-87.6534, 41.8815),  # Lincoln Park
            Point(-87.5967, 41.7897),  # Jackson Park
            Point(-87.5847, 41.7897),  # Washington Park
        ]
        tracts = add_park_access(tracts, major_parks)
        
        # Create placeholder factors for comprehensive index
        rng = np.random.default_rng(0)
        n = len(tracts)
        
        # Create normalized factors (0 to 1)
        rent_norm = (tracts['omega_bar'] - tracts['omega_bar'].min()) / \
                   (tracts['omega_bar'].max() - tracts['omega_bar'].min() + 1e-10)
        owner_norm = rng.uniform(0, 1, size=n)  # Placeholder owner ratio
        income_norm = rng.uniform(0, 1, size=n)  # Placeholder income
        school_norm = rng.uniform(0, 1, size=n)  # Placeholder school quality
        # Transit: use distance (inverse = better)
        transit_proximity = 1.0 / (tracts['transit_distance'] + 0.1)
        transit_norm = (transit_proximity - transit_proximity.min()) / \
                      (transit_proximity.max() - transit_proximity.min() + 1e-10)
        # Park: use distance (inverse = better)
        park_proximity = 1.0 / (tracts['park_access'] + 0.1)
        park_norm = (park_proximity - park_proximity.min()) / \
                   (park_proximity.max() - park_proximity.min() + 1e-10)
        crime_norm = rng.uniform(0, 1, size=n)  # Placeholder crime (inverse)
        
        # Combine with weights
        weights = {
            'rent': 0.15,
            'owner': 0.10,
            'income': 0.20,
            'school': 0.20,
            'transit': 0.15,
            'park': 0.10,
            'crime': 0.10
        }
        
        tracts['v_n'] = (
            rent_norm * weights['rent'] +
            owner_norm * weights['owner'] +
            income_norm * weights['income'] +
            school_norm * weights['school'] +
            transit_norm * weights['transit'] +
            park_norm * weights['park'] +
            crime_norm * weights['crime']
        ) - 0.5

    return tracts


# ================================
# Add placeholder tract attributes
# ================================

def add_placeholder_attributes(tracts: gpd.GeoDataFrame,
                               seed: int = 0) -> gpd.GeoDataFrame:
    """Add omega_bar, K, and v_n using random plausible values."""
    tracts = tracts.copy()
    rng = np.random.default_rng(seed)
    n = len(tracts)

    # Monthly baseline rent (USD)
    tracts["omega_bar"] = rng.uniform(800, 2500, size=n)

    # Tract household capacity
    tracts["K"] = rng.integers(200, 800, size=n)

    # Add park access for comprehensive amenity calculation
    major_parks = [
        Point(-87.6244, 41.8781),  # Millennium Park
        Point(-87.6334, 41.8781),  # Grant Park
        Point(-87.6534, 41.8815),  # Lincoln Park
        Point(-87.5967, 41.7897),  # Jackson Park
        Point(-87.5847, 41.7897),  # Washington Park
    ]
    tracts = add_park_access(tracts, major_parks)
    
    # Add transit access for comprehensive amenity calculation
    major_transit_hubs = [
        Point(-87.6248, 41.8789),  # State/Lake (Loop)
        Point(-87.6327, 41.8789),  # Washington/Wabash (Loop)
        Point(-87.6277, 41.8789),  # Clark/Lake (Loop)
        Point(-87.6534, 41.8815),  # Fullerton (Red/Brown)
        Point(-87.6681, 41.9097),  # Belmont (Red/Brown)
        Point(-87.6552, 41.9037),  # Diversey (Brown)
    ]
    tracts = add_transit_access(tracts, major_transit_hubs)
    
    # Calculate comprehensive amenity index
    tracts = calculate_comprehensive_amenity_index(tracts, use_real_factors=False)

    return tracts


# ================================
# Add initial group shares
# ================================

def add_initial_group_shares(tracts: gpd.GeoDataFrame,
                             seed: int = 0) -> gpd.GeoDataFrame:
    """
    Add initial group shares s_B0, s_W0, s_A0, s_H0 for each tract.
    
    These represent the initial proportion of households in each group:
    - s_B0: Share of group B (Black households)
    - s_W0: Share of group W (White households)  
    - s_A0: Share of group A (Asian households)
    - s_H0: Share of group H (Hispanic/Latino households)
    
    Shares sum to 1.0 for each tract.
    """
    tracts = tracts.copy()
    rng = np.random.default_rng(seed)
    n = len(tracts)
    
    # Generate random proportions for each group (4 groups now)
    # Using Dirichlet distribution to ensure they sum to 1
    alpha = np.array([1.0, 1.0, 1.0, 1.0])  # Equal concentration parameter for 4 groups
    shares = rng.dirichlet(alpha, size=n)
    
    tracts["s_B0"] = shares[:, 0]  # Group B share
    tracts["s_W0"] = shares[:, 1]  # Group W share
    tracts["s_A0"] = shares[:, 2]  # Group A share
    tracts["s_H0"] = shares[:, 3]  # Group H share
    
    # Verify they sum to 1 (with small tolerance for floating point)
    assert np.allclose(tracts[["s_B0", "s_W0", "s_A0", "s_H0"]].sum(axis=1), 1.0), \
        "Group shares must sum to 1.0 for each tract"
    
    return tracts


# ================================
# Standardize column names for model
# ================================

def standardize_column_names(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardize column names to match model notation:
    - omega_bar -> p0 (median rent)
    - v_n -> amenity (amenity index)
    - Keep: K, dw, tract_id, geometry
    - Keep: s_B0, s_W0, s_A0, s_H0
    """
    tracts = tracts.copy()
    
    # Rename columns to match model notation
    rename_dict = {
        "omega_bar": "p0",
        "v_n": "amenity"
    }
    
    # Only rename if columns exist
    for old_name, new_name in rename_dict.items():
        if old_name in tracts.columns:
            tracts = tracts.rename(columns={old_name: new_name})

    return tracts


# ================================
# Main execution
# ================================

if __name__ == "__main__":
    # Configuration: Set to True to use real census data, False for placeholders
    USE_REAL_DATA = True

    # Load tract shapes
    tracts = load_chicago_tracts()

    # Compute commuting distance
    tracts = add_commute_cost(tracts)

    # Fetch real census data or use placeholders
    if USE_REAL_DATA:
        # Fetch real data (includes attributes and group shares)
        tracts = fetch_real_census_data(tracts, use_real_data=True)
    else:
        # Use placeholder data
        tracts = add_placeholder_attributes(tracts)
        # Add initial group shares s_B0, s_W0, s_A0, s_H0
        tracts = add_initial_group_shares(tracts)
    
    # Standardize column names to match model notation
    tracts = standardize_column_names(tracts)
    
    # Filter out non-residential tracts (K=0 or K is NaN)
    # These are typically industrial, commercial, parks, airports, etc.
    initial_count = len(tracts)
    tracts = tracts[(tracts['K'] > 0) & (tracts['K'].notna())].copy()
    filtered_count = initial_count - len(tracts)
    
    if filtered_count > 0:
        print(f"\nFiltered out {filtered_count} non-residential tracts (K=0 or missing)")
        print("These are typically industrial, commercial, parks, airports, or other non-residential areas.")

    # Show resulting structure
    print("\n" + "="*60)
    print("Chicago Census Tracts with Initial Conditions")
    print("="*60)
    print(f"\nTotal tracts (residential only): {len(tracts)}")
    print(f"\nColumns: {list(tracts.columns)}")
    print("\nFirst few rows:")
    print(tracts.head())
    print("\nSummary statistics - Model Parameters:")
    print(tracts[['p0', 'K', 'dw', 'amenity']].describe())
    print("\nSummary statistics - Initial Group Shares:")
    print(tracts[['s_B0', 's_W0', 's_A0', 's_H0']].describe())
    print(f"\nVerification - Group shares sum to 1.0: {np.allclose(tracts[['s_B0', 's_W0', 's_A0', 's_H0']].sum(axis=1), 1.0)}")
    
    # Save the enriched shapefile
    output_path = Path("data") / "chicago_tracts_enriched.shp"
    tracts.to_file(output_path)
    print(f"\nSaved enriched tracts to: {output_path}")
    
    # Visualize maps
    import matplotlib.pyplot as plt
    
    # Create output directory for figures
    fig_dir = Path("data") / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Get city boundary to set plot limits (excludes lake)
    city_boundary = get_city_boundary(tracts)
    bounds = city_boundary.total_bounds  # [minx, miny, maxx, maxy]
    
    # Helper function to set plot bounds to city boundary only
    def set_plot_bounds(fig, ax, bounds):
        """Set plot limits to exactly match city bounds."""
        # Set plot limits to exactly match city bounds (no padding to avoid showing lake)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        # Set equal aspect to prevent distortion
        ax.set_aspect('equal', adjustable='box')
    
    # Helper function to plot with single color gradient
    def plot_single_gradient(tracts, column, title, cmap, fig_dir, filename, bounds):
        """Plot a single variable with a continuous color gradient."""
        # Create a copy and fill NaN values with median for visualization
        tracts_plot = tracts.copy()
        if tracts_plot[column].isna().any():
            tracts_plot[column] = tracts_plot[column].fillna(tracts_plot[column].median())
        
        fig, ax = plt.subplots(figsize=(12, 10))
        # Plot with single continuous colormap
        tracts_plot.plot(column=column, legend=True, ax=ax, cmap=cmap, 
                        edgecolor='#777777', linewidth=0.25, missing_kwds={'color': 'lightgray'})
        # Draw tract outlines clearly on top of the fill so boundaries are visible
        tracts_plot.boundary.plot(ax=ax, color='#2f2f2f', linewidth=0.6, alpha=0.9)
        set_plot_bounds(fig, ax, bounds)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # 1. Commuting distance map
    print("\nCreating commute distance map...")
    plot_single_gradient(tracts, "dw", "Commuting Distance to Downtown Chicago (km)", 
                        "viridis", fig_dir, "chicago_commute_distance.png", bounds)
    print(f"Saved commute distance map to: {fig_dir / 'chicago_commute_distance.png'}")
    
    # 2. Baseline rent map (p0)
    print("Creating baseline rent map...")
    plot_single_gradient(tracts, "p0", "Median Rent (p0) - USD per month", 
                        "YlOrRd", fig_dir, "chicago_baseline_rent.png", bounds)
    print(f"Saved baseline rent map to: {fig_dir / 'chicago_baseline_rent.png'}")
    
    # 3. Capacity map
    print("Creating capacity map...")
    plot_single_gradient(tracts, "K", "Household Capacity (K)", 
                        "Blues", fig_dir, "chicago_capacity.png", bounds)
    print(f"Saved capacity map to: {fig_dir / 'chicago_capacity.png'}")
    
    # 4. Amenity index map
    print("Creating amenity index map...")
    plot_single_gradient(tracts, "amenity", "Amenity Index", 
                        "RdYlGn", fig_dir, "chicago_amenity_index.png", bounds)
    print(f"Saved amenity index map to: {fig_dir / 'chicago_amenity_index.png'}")
    
    # 5. Initial group shares maps
    print("Creating initial group share maps...")
    for group, col in [('B', 's_B0'), ('W', 's_W0'), ('A', 's_A0'), ('H', 's_H0')]:
        plot_single_gradient(tracts, col, f"Initial Share of Group {group} (s_{group}0)", 
                            "Reds", fig_dir, f"chicago_group_{group}_share.png", bounds)
        print(f"Saved group {group} share map to: {fig_dir / f'chicago_group_{group}_share.png'}")
    
    print("\nAll static maps saved successfully!")
    
    # Create interactive map with folium
    try:
        import folium
        from folium import plugins
        
        print("\nCreating interactive map...")
        
        # Calculate center of Chicago
        center_lat = tracts.geometry.centroid.y.mean()
        center_lon = tracts.geometry.centroid.x.mean()
        
        # Get bounds first (before creating map) to set max_bounds
        tracts_wgs84_temp = tracts.to_crs(epsg=4326)
        bounds_temp = tracts_wgs84_temp.total_bounds
        padding_temp = 0.02  # Small padding for max_bounds
        
        # Create base map with max_bounds to restrict panning
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap',
            max_bounds=[[bounds_temp[1] - padding_temp, bounds_temp[0] - padding_temp],
                        [bounds_temp[3] + padding_temp, bounds_temp[2] + padding_temp]]
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Ensure tracts are in WGS84 for folium
        tracts_wgs84 = tracts.to_crs(epsg=4326)
        
        # Fill NaN values with median for each column to ensure continuous gradients
        for col in ['dw', 'p0', 'K', 'amenity', 's_B0', 's_W0', 's_A0', 's_H0']:
            if col in tracts_wgs84.columns and tracts_wgs84[col].isna().any():
                tracts_wgs84[col] = tracts_wgs84[col].fillna(tracts_wgs84[col].median())
        
        # Get bounds from tracts to set map view to city only
        bounds = tracts_wgs84.total_bounds  # [minx, miny, maxx, maxy]
        
        # Set map bounds to city boundary (excludes lake)
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
        # Create feature groups for each attribute layer
        import branca.colormap as cm
        
        # 1. Commuting Distance layer
        commute_layer = folium.FeatureGroup(name='Commute Distance', show=True)
        # Use viridis colormap (continuous gradient from dark purple to yellow)
        dw_min, dw_max = tracts_wgs84['dw'].min(), tracts_wgs84['dw'].max()
        dw_colormap = cm.LinearColormap(
            colors=['#440154', '#482777', '#3f4a8a', '#31688e', '#26838f', '#1f9d8a', 
                   '#6cce5a', '#b6de2b', '#fee825'],
            vmin=dw_min,
            vmax=dw_max
        )
        
        for idx, row in tracts_wgs84.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda feature, r=row: {
                    'fillColor': dw_colormap(r['dw']),
                    'color': 'gray',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.Tooltip(
                    f"<b>Tract:</b> {row['tract_id']}<br>"
                    f"<b>Commute Distance:</b> {row['dw']:.2f} km",
                    sticky=True
                )
            ).add_to(commute_layer)
        commute_layer.add_to(m)
        dw_colormap.add_to(m)
        
        # 2. Median Rent (p0) layer
        rent_layer = folium.FeatureGroup(name='Median Rent (p0)', show=False)
        # Use YlOrRd colormap (continuous gradient from yellow to red)
        p0_min, p0_max = tracts_wgs84['p0'].min(), tracts_wgs84['p0'].max()
        rent_colormap = cm.LinearColormap(
            colors=['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026'],
            vmin=p0_min,
            vmax=p0_max
        )
        
        for idx, row in tracts_wgs84.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda feature, r=row: {
                    'fillColor': rent_colormap(r['p0']),
                    'color': 'gray',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.Tooltip(
                    f"<b>Tract:</b> {row['tract_id']}<br>"
                    f"<b>Median Rent (p0):</b> ${row['p0']:.0f}/month",
                    sticky=True
                )
            ).add_to(rent_layer)
        rent_layer.add_to(m)
        
        # 3. Capacity layer
        capacity_layer = folium.FeatureGroup(name='Capacity', show=False)
        # Use Blues colormap (continuous gradient from light to dark blue)
        k_min, k_max = tracts_wgs84['K'].min(), tracts_wgs84['K'].max()
        k_colormap = cm.LinearColormap(
            colors=['#eff3ff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#084594'],
            vmin=k_min,
            vmax=k_max
        )
        
        for idx, row in tracts_wgs84.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda feature, r=row: {
                    'fillColor': k_colormap(r['K']),
                    'color': 'gray',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.Tooltip(
                    f"<b>Tract:</b> {row['tract_id']}<br>"
                    f"<b>Capacity:</b> {row['K']}",
                    sticky=True
                )
            ).add_to(capacity_layer)
        capacity_layer.add_to(m)
        
        # 4. Amenity Index layer
        amenity_layer = folium.FeatureGroup(name='Amenity Index', show=False)
        # Use RdYlGn colormap (continuous gradient from red through yellow to green)
        amenity_min, amenity_max = tracts_wgs84['amenity'].min(), tracts_wgs84['amenity'].max()
        v_colormap = cm.LinearColormap(
            colors=['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#1a9850'],
            vmin=amenity_min,
            vmax=amenity_max
        )
        
        for idx, row in tracts_wgs84.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda feature, r=row: {
                    'fillColor': v_colormap(r['amenity']),
                    'color': 'gray',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=folium.Tooltip(
                    f"<b>Tract:</b> {row['tract_id']}<br>"
                    f"<b>Amenity Index:</b> {row['amenity']:.3f}",
                    sticky=True
                )
            ).add_to(amenity_layer)
        amenity_layer.add_to(m)
        
        # 5. Initial Group Share layers
        for group, col, group_name in [('B', 's_B0', 'Group B'), ('W', 's_W0', 'Group W'), ('A', 's_A0', 'Group A'), ('H', 's_H0', 'Group H (Hispanic)')]:
            share_layer = folium.FeatureGroup(name=f'Initial Share {group_name} (s_{group}0)', show=False)
            # Use Reds colormap (continuous gradient from light to dark red)
            share_colormap = cm.LinearColormap(
                colors=['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
                vmin=0.0,
                vmax=1.0
            )
            
            for idx, row in tracts_wgs84.iterrows():
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda feature, r=row, c=col: {
                        'fillColor': share_colormap(r[c]),
                        'color': 'gray',
                        'weight': 0.5,
                        'fillOpacity': 0.7,
                    },
                    tooltip=folium.Tooltip(
                        f"<b>Tract:</b> {row['tract_id']}<br>"
                        f"<b>Initial Share {group_name}:</b> {row[col]:.3f}",
                        sticky=True
                    )
                ).add_to(share_layer)
            share_layer.add_to(m)
        
        # Add a layer with all attributes in tooltip
        info_layer = folium.FeatureGroup(name='All Attributes (Info)', show=False)
        for idx, row in tracts_wgs84.iterrows():
            tooltip_text = f"""
            <div style="width: 250px;">
            <b>Tract ID:</b> {row['tract_id']}<br>
            <b>Commute Distance (dw):</b> {row['dw']:.2f} km<br>
            <b>Median Rent (p0):</b> ${row['p0']:.0f}/month<br>
            <b>Capacity (K):</b> {row['K']}<br>
            <b>Amenity Index:</b> {row['amenity']:.3f}<br>
            <b>Initial Share B (s_B0):</b> {row['s_B0']:.3f}<br>
            <b>Initial Share W (s_W0):</b> {row['s_W0']:.3f}<br>
            <b>Initial Share A (s_A0):</b> {row['s_A0']:.3f}<br>
            <b>Initial Share H (s_H0):</b> {row['s_H0']:.3f}
            </div>
            """
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda feature: {
                    'fillColor': '#f0f0f0',  # Light gray fill so tracts are visible
                    'color': '#333333',  # Dark gray border
                    'weight': 1.5,
                    'fillOpacity': 0.6,  # Semi-transparent so map tiles are still visible
                },
                tooltip=folium.Tooltip(tooltip_text, sticky=True)
            ).add_to(info_layer)
        info_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save interactive map
        map_path = fig_dir / "chicago_tracts_interactive.html"
        m.save(str(map_path))
        print(f"Saved interactive map to: {map_path}")
        print(f"\nOpen the map in your browser: {map_path.absolute()}")
        
    except ImportError:
        print("\nNote: folium not installed. Install with: pip install folium")
        print("Skipping interactive map creation.")
