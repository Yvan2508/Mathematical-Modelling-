"""
Animated Visualization of Agent Movement
----------------------------------------
This module creates animated visualizations showing agents (households) moving
across tracts until they settle in their final locations.

Supports:
- Interactive HTML animation (Folium with time slider)
- Video/GIF animation (Matplotlib)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import json


# ==============================
# Scale tract capacity for simulation
# ==============================

def scale_tract_capacity(tracts: gpd.GeoDataFrame,
                         N_agents: int,
                         target_occupancy: float = 0.95) -> gpd.GeoDataFrame:
    """
    Scale tract capacities to match simulation size while maintaining realistic occupancy.

    This ensures that with fewer agents than the real population, we still have
    realistic housing constraints and occupancy rates.

    Parameters
    ----------
    tracts : GeoDataFrame
        Tracts with 'K' (capacity) column
    N_agents : int
        Number of agents in simulation
    target_occupancy : float
        Desired occupancy rate (default 0.95 = 95% occupied)
        Typical real-world occupancy is 90-95%

    Returns
    -------
    GeoDataFrame with scaled 'K' values and original K saved as 'K_original'

    Example
    -------
    If real Chicago has 1M housing units and you simulate 50K agents (5% of real):
    - Original K per tract is scaled down by 5%
    - Target occupancy ensures 95% of scaled capacity is filled
    - Movement constraints remain realistic at simulation scale
    """
    tracts = tracts.copy()

    # Calculate scaling factor
    total_real_capacity = tracts['K'].sum()

    # Target capacity = N_agents / target_occupancy
    # (so when N_agents fill it, occupancy = target_occupancy)
    target_total_capacity = N_agents / target_occupancy

    scaling_factor = target_total_capacity / total_real_capacity

    # Scale each tract's capacity proportionally
    tracts['K_original'] = tracts['K']  # Keep original for reference
    tracts['K'] = np.maximum(1, np.round(tracts['K'] * scaling_factor)).astype(int)

    # Calculate actual occupancy after scaling
    actual_occupancy = N_agents / tracts['K'].sum()

    print("\n" + "=" * 60)
    print("CAPACITY SCALING FOR SIMULATION")
    print("=" * 60)
    print(f"Original total capacity:     {total_real_capacity:>12,.0f} housing units")
    print(f"Scaled total capacity:       {tracts['K'].sum():>12,.0f} housing units")
    print(f"Number of agents:            {N_agents:>12,.0f} households")
    print(f"Target occupancy rate:       {target_occupancy:>12.1%}")
    print(f"Actual occupancy rate:       {actual_occupancy:>12.1%}")
    print(f"Scaling factor:              {scaling_factor:>12.4f}")
    print(f"\nThis maintains realistic housing constraints at simulation scale.")
    print("=" * 60 + "\n")

    return tracts



# ==============================
# Run simulation using real dynamics
# ==============================

def run_simulation_with_agent_history(
    agents: pd.DataFrame,
    tracts: gpd.GeoDataFrame,
    T: int = 20,
    seed: int = 0,
    use_base_model: bool = False,
    tolerance=None,
) -> list:
    """
    Run the full simulation and return agent history for animation.
    
    Parameters
    ----------
    agents : DataFrame
        Initial agent locations
    tracts : GeoDataFrame
        Tracts with geometry and attributes
    T : int
        Number of time periods to simulate
    seed : int
        Random seed
    use_base_model : bool
        If True, use the simplified base model (only own-group share preferences).
        If False, use the full dynamics model.
    tolerance : float or str, optional
        Only used when use_base_model=True. Own-group share threshold
        (e.g., 'very_high', 'medium', 'very_low' or a custom float).
    
    Returns
    -------
    agent_history : list
        List of DataFrames, one per time step, showing agent locations
    """
    import numpy as np
    if use_base_model:
        from base_model import one_period_update_base
    else:
        from dynamics import one_period_update
    
    rng = np.random.default_rng(seed)
    
    # Get initial rents (ignored in base model)
    if use_base_model:
        omega_t = np.zeros(len(tracts), dtype=float)
    else:
        if "p0" in tracts.columns:
            omega_t = tracts["p0"].values.copy()
        elif "omega_bar" in tracts.columns:
            omega_t = tracts["omega_bar"].values.copy()
        else:
            raise ValueError("Tracts must have 'p0' or 'omega_bar' column")
    
    # Store agent history (one DataFrame per time step)
    agent_history = [agents.copy()]
    
    # Run simulation period by period
    for t in range(T):
        print(f"Simulating period {t+1}/{T}...", end=" ", flush=True)
        if use_base_model:
            tol_arg = tolerance if tolerance is not None else "medium"
            agents, omega_t, H_n, H_ng, s_ng = one_period_update_base(
                agents, tracts, omega_t, rng, tolerance=tol_arg
            )
        else:
            agents, omega_t, H_n, H_ng, s_ng = one_period_update(
                agents, tracts, omega_t, rng
            )
        # Store agent state at this time step
        agent_history.append(agents.copy())
        print(f"✓ ({len(agent_history)} steps stored)", flush=True)
    
    print(f"\nSimulation complete! Total steps: {len(agent_history)}")
    return agent_history


# ==============================
# Calculate tract centroids for agent positioning
# ==============================

def get_tract_centroids(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get centroids of all tracts for positioning agents."""
    tracts_metric = tracts.to_crs(epsg=26916)  # UTM for accurate centroids
    centroids = tracts_metric.geometry.centroid
    centroids_wgs84 = centroids.to_crs(epsg=4326)
    
    centroids_df = gpd.GeoDataFrame({
        'tract_id': tracts['tract_id'].values,
        'tract_idx': tracts.index.values,
        'centroid_lon': centroids_wgs84.x,
        'centroid_lat': centroids_wgs84.y,
    }, geometry=centroids_wgs84)
    
    return centroids_df


# ==============================
# Create animated HTML map (Folium)
# ==============================

def create_animated_html_map(history: list,
                              tracts: gpd.GeoDataFrame,
                              output_path: Path,
                              sample_rate: int = 5) -> None:
    """
    Create an interactive HTML map with time slider showing agent movement.
    
    Parameters
    ----------
    history : list
        List of agent DataFrames, one per time step
    tracts : GeoDataFrame
        Tracts with geometry
    output_path : Path
        Where to save the HTML file
    sample_rate : int
        Show every Nth frame (to reduce file size)
    """
    # Sample history (every Nth step)
    sampled_history = history[::sample_rate]
    if len(sampled_history) == 0:
        sampled_history = history
    
    # Get tract centroids
    centroids = get_tract_centroids(tracts)
    
    # Calculate map center
    center_lat = tracts.geometry.centroid.y.mean()
    center_lon = tracts.geometry.centroid.x.mean()
    
    group_colors = {'B': '#e74c3c', 'W': '#3498db', 'A': '#2ecc71', 'H': '#f39c12'}
    job_centers = [
        {"name": "Loop", "lat": 41.8781, "lon": -87.6298},
        {"name": "O'Hare", "lat": 41.9810, "lon": -87.9090},
        {"name": "Elk Grove", "lat": 41.9900, "lon": -87.9700},
        {"name": "Calumet", "lat": 41.6900, "lon": -87.5700},
    ]
    # Preserve tract outlines (GeoJSON) so geography stays visible in the map background
    tracts_geojson = json.loads(
        tracts.to_crs(epsg=4326)[['tract_id', 'geometry']].to_json()
    )
    
    # Collect step data for JavaScript animation
    step_data = []
    
    for step, agents in enumerate(sampled_history):
        # Count agents per tract for this step
        tract_counts = agents.groupby('tract_idx').size()
        tract_groups = agents.groupby(['tract_idx', 'group']).size().unstack(fill_value=0)
        
        step_markers = []
        for tract_idx in tract_counts.index:
            # Get centroid
            centroid_row = centroids[centroids['tract_idx'] == tract_idx]
            if len(centroid_row) == 0:
                continue
            
            lon, lat = centroid_row.iloc[0]['centroid_lon'], centroid_row.iloc[0]['centroid_lat']
            
            # Count agents by group in this tract
            if tract_idx in tract_groups.index:
                group_counts = tract_groups.loc[tract_idx].to_dict()
            else:
                group_counts = {'B': 0, 'W': 0, 'A': 0, 'H': 0}
            
            total_agents = tract_counts[tract_idx]
            
            # Determine color based on majority group
            majority_group = max(group_counts, key=group_counts.get) if total_agents > 0 else 'W'
            color = group_colors.get(majority_group, '#95a5a6')
            
            # Marker size proportional to agent count
            radius = max(5, min(30, np.sqrt(total_agents) * 3))
            
            step_markers.append({
                'tract_id': str(tracts.iloc[tract_idx]['tract_id']),
                'tract_idx': int(tract_idx),
                'lat': float(lat),
                'lon': float(lon),
                'total_agents': int(total_agents),
                'group_counts': {k: int(v) for k, v in group_counts.items()},
                'color': color,
                'radius': float(radius)
            })
        
        step_data.append(step_markers)
    
    num_steps = len(sampled_history)
    num_steps_minus_one = max(num_steps - 1, 0)
    
    # Embed step data as JSON
    step_data_json = json.dumps(step_data)
    tracts_geojson_json = json.dumps(tracts_geojson)
    job_centers_json = json.dumps(job_centers)
    
    # Create standalone HTML file with Leaflet.js
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Movement Animation</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
    <style>
        html, body {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        #animation-controls {{
            position: fixed;
            top: 10px;
            right: 10px;
            width: 240px;
            background-color: white;
            z-index: 1000;
            font-size: 14px;
            border: 2px solid #333;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            pointer-events: auto;
        }}
        #legend {{
            position: fixed;
            bottom: 50px;
            right: 50px;
            width: 200px;
            background-color: white;
            z-index: 1000;
            font-size: 14px;
            border: 2px solid grey;
            border-radius: 5px;
            padding: 10px;
        }}
        .legend-dot {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .legend-job {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 6px;
            background: #8e44ad;
            border: 1px solid #2c2050;
        }}
        button {{
            flex: 1;
            padding: 8px;
            cursor: pointer;
            border: 1px solid #333;
            border-radius: 3px;
            background: #f0f0f0;
            pointer-events: auto;
        }}
        #play-btn {{
            background: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        .button-group {{
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }}
        .pie-marker {{
            background: transparent;
            border: none;
        }}
        .pie-marker .pie {{
            border-radius: 50%;
            border: 1px solid #333;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div id="animation-controls">
        <h4 style="margin-top: 0;">Animation Controls</h4>
        <div id="step-indicator" style="margin: 4px 0 10px 0; font-weight: bold; color: #333;">
            t = 0 / {num_steps_minus_one}
        </div>
        <div class="button-group">
            <button id="prev-btn">◀ Prev</button>
            <button id="play-btn">Play</button>
            <button id="next-btn">Next ▶</button>
        </div>
        <div style="margin-top: 15px; margin-bottom: 10px;">
            <label for="map-theme-select" style="display: block; font-size: 12px; font-weight: bold; margin-bottom: 5px; color: #333;">Map Theme:</label>
            <select id="map-theme-select" style="width: 100%; padding: 6px; border: 1px solid #333; border-radius: 3px; font-size: 13px; background: white; cursor: pointer;">
                <option value="Light (CartoDB)" selected>Light (CartoDB)</option>
                <option value="OpenStreetMap">OpenStreetMap</option>
                <option value="Dark (CartoDB)">Dark (CartoDB)</option>
                <option value="Terrain (Stamen)">Terrain (Stamen)</option>
                <option value="Toner (Stamen)">Toner (Stamen)</option>
            </select>
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 10px;">
            <p>✓ Animation auto-plays on load</p>
            <p>Click Play/Pause to control</p>
        </div>
    </div>
    
    <div id="legend">
        <h4>Agent Groups</h4>
        <p><span class="legend-dot" style="background:#e74c3c"></span> Black (B)</p>
        <p><span class="legend-dot" style="background:#3498db"></span> White (W)</p>
        <p><span class="legend-dot" style="background:#2ecc71"></span> Asian (A)</p>
        <p><span class="legend-dot" style="background:#f39c12"></span> Hispanic (H)</p>
        <hr>
        <p><span class="legend-job"></span> Job Center</p>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script>
        // Configuration
        var centerLat = {center_lat};
        var centerLon = {center_lon};
        var stepData = {step_data_json};
        var numSteps = {num_steps};
        var tractsGeoJson = {tracts_geojson_json};
        var jobCenters = {job_centers_json};
        
        // Animation state
        var currentStep = 0;
        var isPlaying = false;
        var intervalId = null;
        var speed = 500;
        var groupColors = {{
            B: '#e74c3c',
            W: '#3498db',
            A: '#2ecc71',
            H: '#f39c12'
        }};
        var groupOrder = ['B', 'W', 'A', 'H'];
        
        // Map and layers
        var map = null;
        var markerLayer = null;
        var jobCenterLayer = null;
        var baseLayers = {{}};
        
        // Initialize map
        function initMap() {{
            // Create map
            map = L.map('map', {{
                center: [centerLat, centerLon],
                zoom: 10
            }});
            
            // Create marker layer
            markerLayer = L.layerGroup().addTo(map);
            jobCenterLayer = L.layerGroup().addTo(map);
            
            // Add tile layers
            baseLayers['OpenStreetMap'] = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19
            }});
            
            baseLayers['Light (CartoDB)'] = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 20
            }});
            
            baseLayers['Dark (CartoDB)'] = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 20
            }});
            
            baseLayers['Terrain (Stamen)'] = L.tileLayer('https://stamen-tiles-{{s}}.a.ssl.fastly.net/terrain/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                subdomains: 'abcd',
                maxZoom: 18
            }});
            
            baseLayers['Toner (Stamen)'] = L.tileLayer('https://stamen-tiles-{{s}}.a.ssl.fastly.net/toner/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                subdomains: 'abcd',
                maxZoom: 18
            }});
            
            // Add default layer
            baseLayers['Light (CartoDB)'].addTo(map);
            window.currentBaseLayer = baseLayers['Light (CartoDB)'];

            // Keep tract boundaries visible underneath the moving agents
            map.createPane('tractsPane');
            map.getPane('tractsPane').style.zIndex = 200;
            map.getPane('tractsPane').style.pointerEvents = 'none';
            L.geoJSON(tractsGeoJson, {{
                pane: 'tractsPane',
                style: function(feature) {{
                    return {{
                        color: '#444',
                        weight: 1,
                        opacity: 0.8,
                        fillOpacity: 0.03
                    }};
                }}
            }}).addTo(map);

            // Add job center markers
            var centerIcon = L.divIcon({{
                className: 'job-center-icon',
                html: '<div style="width:16px;height:16px;background:#8e44ad;border:2px solid #2c2050;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,0.35);"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            }});

            jobCenters.forEach(function(c) {{
                L.marker([c.lat, c.lon], {{ icon: centerIcon }})
                    .bindTooltip('Job Center: ' + c.name, {{permanent: false, direction: 'top'}})
                    .addTo(jobCenterLayer);
            }});
        }}
        
        // Switch map theme
        function switchMapTheme(themeName) {{
            if (!map || !baseLayers[themeName]) return;
            
            // Remove current base layer
            if (window.currentBaseLayer) {{
                map.removeLayer(window.currentBaseLayer);
            }}
            
            // Add new base layer
            baseLayers[themeName].addTo(map);
            window.currentBaseLayer = baseLayers[themeName];
        }}
        
        // Show a specific step
        function showStep(step) {{
            if (step < 0 || step >= numSteps) return;
            
            currentStep = step;
            var indicator = document.getElementById('step-indicator');
            if (indicator) {{
                indicator.textContent = 't = ' + step + ' / ' + (numSteps - 1);
            }}
            
            // Clear existing markers
            markerLayer.clearLayers();
            
            // Get markers for this step
            var markers = stepData[step];
            if (!markers || markers.length === 0) return;

            // Helper to create a conic-gradient pie from group shares
            function buildPieBackground(groupCounts) {{
                var total = 0;
                for (var i = 0; i < groupOrder.length; i++) {{
                    var g = groupOrder[i];
                    total += groupCounts[g] || 0;
                }}
                if (total === 0) {{
                    return '#cccccc';
                }}
                var segments = [];
                var current = 0;
                for (var j = 0; j < groupOrder.length; j++) {{
                    var group = groupOrder[j];
                    var share = (groupCounts[group] || 0) / total;
                    var start = current * 100;
                    var end = (current + share) * 100;
                    segments.push(groupColors[group] + ' ' + start + '% ' + end + '%');
                    current += share;
                }}
                return 'conic-gradient(' + segments.join(', ') + ')';
            }}
            
            // Create markers
            for (var i = 0; i < markers.length; i++) {{
                var m = markers[i];
                
                // Create popup content
                var popupContent = '<b>Tract: ' + m.tract_id + '</b><br>' +
                    'Total Agents: ' + m.total_agents + '<br>' +
                    'Black (B): ' + (m.group_counts.B || 0) + '<br>' +
                    'White (W): ' + (m.group_counts.W || 0) + '<br>' +
                    'Asian (A): ' + (m.group_counts.A || 0) + '<br>' +
                    'Hispanic (H): ' + (m.group_counts.H || 0);
                
                // Create tooltip
                var totalAgents = m.total_agents || 0;
                function pct(val) {{
                    if (!totalAgents) return '0%';
                    return ((val / totalAgents) * 100).toFixed(1) + '%';
                }}
                var tooltipContent = '<b>Tract ' + m.tract_id + '</b><br>' +
                    'B: ' + pct(m.group_counts.B || 0) + ' &nbsp; ' +
                    'W: ' + pct(m.group_counts.W || 0) + ' &nbsp; ' +
                    'A: ' + pct(m.group_counts.A || 0) + ' &nbsp; ' +
                    'H: ' + pct(m.group_counts.H || 0);
                
                // Build pie-marker icon
                var diameter = m.radius * 2;
                var pieBackground = buildPieBackground(m.group_counts || {{}});
                var iconHtml = '<div class="pie" style="width:' + diameter + 'px;height:' + diameter + 'px;background:' + pieBackground + ';"></div>';
                var icon = L.divIcon({{
                    className: 'pie-marker',
                    html: iconHtml,
                    iconSize: [diameter, diameter],
                    iconAnchor: [diameter / 2, diameter / 2]
                }});
                
                // Create marker with pie chart fill
                var marker = L.marker([m.lat, m.lon], {{ icon: icon }});
                
                marker.bindPopup(popupContent);
                marker.bindTooltip(tooltipContent);
                marker.addTo(markerLayer);
            }}
        }}
        
        // Animation controls
        function toggleAnimation() {{
            if (isPlaying) {{
                // Pause
                if (intervalId) {{
                    clearInterval(intervalId);
                    intervalId = null;
                }}
                isPlaying = false;
                document.getElementById('play-btn').textContent = 'Play';
            }} else {{
                // Play
                intervalId = setInterval(function() {{
                    currentStep = (currentStep + 1) % numSteps;
                    showStep(currentStep);
                }}, speed);
                isPlaying = true;
                document.getElementById('play-btn').textContent = 'Pause';
            }}
        }}
        
        function stepForward() {{
            if (isPlaying) toggleAnimation();
            showStep((currentStep + 1) % numSteps);
        }}
        
        function stepBackward() {{
            if (isPlaying) toggleAnimation();
            showStep((currentStep - 1 + numSteps) % numSteps);
        }}
        
        // Setup button handlers
        function setupButtons() {{
            document.getElementById('prev-btn').onclick = stepBackward;
            document.getElementById('play-btn').onclick = toggleAnimation;
            document.getElementById('next-btn').onclick = stepForward;
            
            // Map theme switcher
            var themeSelect = document.getElementById('map-theme-select');
            if (themeSelect) {{
                themeSelect.onchange = function() {{
                    switchMapTheme(this.value);
                }};
            }}
        }}
        
        // Initialize when page loads
        window.addEventListener('load', function() {{
            initMap();
            setupButtons();
            showStep(0);
            // Auto-start animation after a short delay
            setTimeout(function() {{
                toggleAnimation();
            }}, 1000);
        }});
    </script>
</body>
</html>'''
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved animated HTML map to: {output_path}")
    
    # Open in browser automatically
    import webbrowser
    import os
    import platform
    
    abs_path = os.path.abspath(output_path)
    
    # On macOS, use 'open' command; on other systems use webbrowser
    if platform.system() == 'Darwin':  # macOS
        os.system(f'open "{abs_path}"')
        print(f"Opening animation in default browser...")
    else:
        # Use file:// URL format for local files
        file_url = f"file://{abs_path}"
        webbrowser.open(file_url)
        print(f"Opening animation in browser: {file_url}")


# ==============================
# Create animated video/GIF (Matplotlib)
# ==============================

def create_animated_video(history: list,
                          tracts: gpd.GeoDataFrame,
                          output_path: Path,
                          sample_rate: int = 2,
                          fps: int = 5,
                          dpi: int = 100) -> None:
    """
    Create an animated video/GIF showing agent movement.
    
    Parameters
    ----------
    history : list
        List of agent DataFrames, one per time step
    tracts : GeoDataFrame
        Tracts with geometry
    output_path : Path
        Where to save the video/GIF (.mp4 or .gif)
    sample_rate : int
        Show every Nth frame
    fps : int
        Frames per second
    dpi : int
        Resolution
    """
    # Sample history
    sampled_history = history[::sample_rate]
    if len(sampled_history) == 0:
        sampled_history = history
    
    # Get tract centroids
    centroids = get_tract_centroids(tracts)
    
    # Prepare tracts for plotting
    tracts_wgs84 = tracts.to_crs(epsg=4326)
    bounds = tracts_wgs84.total_bounds
    
    # Group colors
    group_colors = {'B': '#e74c3c', 'W': '#3498db', 'A': '#2ecc71', 'H': '#f39c12'}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    def animate(frame):
        ax.clear()
        
        # Plot tracts
        tracts_wgs84.plot(ax=ax, color='#f2f2f2', edgecolor='#bdbdbd', linewidth=0.4, alpha=0.9)
        tracts_wgs84.boundary.plot(ax=ax, color='#4a4a4a', linewidth=0.7, alpha=0.9)
        
        # Get agents for this frame
        agents = sampled_history[frame]
        
        # Plot agents by group
        for group, color in group_colors.items():
            group_agents = agents[agents['group'] == group]
            if len(group_agents) == 0:
                continue
            
            # Get centroids for agent tracts
            agent_centroids = centroids[centroids['tract_idx'].isin(group_agents['tract_idx'])]
            
            if len(agent_centroids) > 0:
                ax.scatter(agent_centroids['centroid_lon'], 
                          agent_centroids['centroid_lat'],
                          c=color, s=20, alpha=0.6, label=group, edgecolors='black', linewidths=0.3)
        
        # Set bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Agent Movement - Step {frame}/{len(sampled_history)-1}', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        if frame == 0:  # Only add legend once
            ax.legend(loc='upper right', title='Groups', framealpha=0.9)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(sampled_history), 
                        interval=1000/fps, repeat=True, blit=False)
    
    # Save animation
    if output_path.suffix == '.gif':
        anim.save(str(output_path), writer='pillow', fps=fps, dpi=dpi)
    else:
        anim.save(str(output_path), writer='ffmpeg', fps=fps, dpi=dpi)
    
    plt.close()
    print(f"Saved animated video to: {output_path}")


# ==============================
# Main execution
# ==============================

if __name__ == "__main__":
    # Import required modules
    from agents import initialize_agents
    
    # Load tracts
    tracts_path = Path("data") / "chicago_tracts_enriched.shp"
    if not tracts_path.exists():
        print(f"Error: Tracts file not found at {tracts_path}")
        print("Please run prepare_tract.py first.")
    else:
        print("Loading tracts...")
        tracts = gpd.read_file(tracts_path)
        

        # ====================================================
        # SIMULATION CONFIGURATION
        # ====================================================
        N_agents = 50_000  # Number of agents to simulate (scaled down from real population)
        target_occupancy = 0.95  # Target 95% occupancy rate (realistic)
        T_periods = 100  # Number of simulation periods
        use_base_model = False  # True = simple model (only group preferences), False = full model

        print(f"\n{'=' * 60}")
        print("SIMULATION CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Number of agents:            {N_agents:>12,}")
        print(f"Target occupancy rate:       {target_occupancy:>12.1%}")
        print(f"Simulation periods:          {T_periods:>12,}")
        print(f"Model type:                  {'Base (simplified)' if use_base_model else 'Full (realistic)'}")
        print(f"{'=' * 60}\n")

        # Scale tract capacities to match simulation size
        tracts = scale_tract_capacity(tracts, N_agents, target_occupancy=target_occupancy)

        # Initialize agents with scaled tracts
        print("Initializing agents...")
        agents = initialize_agents(
            tracts,
            N_households=N_agents,
            use_tract_shares=True,
            seed=42
        )
        print(f"Initialized {len(agents)} agents (households)")

        # Verify occupancy
        actual_occupancy = len(agents) / tracts['K'].sum()
        print(f"Actual initial occupancy: {actual_occupancy:.1%}")

        # Run simulation
        print("\nRunning simulation with real dynamics..." if not use_base_model else "\nRunning simulation with base model...")
        print("=" * 60)
        T_periods = 100  # 42 periods = 43 steps total (initial state + 42 periods)
        print(f"Simulating {T_periods} periods with {len(agents)} agents...")
        print("This may take a few minutes...\n")
        
        agent_history = run_simulation_with_agent_history(
            agents, 
            tracts, 
            T=T_periods, 
            seed=123,
            use_base_model=use_base_model,
        )
        print("=" * 60)
        print(f"\nSimulation complete! Generated {len(agent_history)} time steps")
        
        # Create output directory
        output_dir = Path("data") / "figures"
        output_dir.mkdir(exist_ok=True)
        
        # Create HTML animation
        print("\nCreating HTML animation...")
        html_path = output_dir / "agent_movement_animation.html"
        # Sample every 2nd step to reduce file size (or use sample_rate=1 for all steps)
        create_animated_html_map(agent_history, tracts, html_path, sample_rate=1)
        
        # Create video animation (optional - requires ffmpeg or pillow)
        try:
            print("Creating video animation...")
            video_path = output_dir / "agent_movement_animation.gif"
            create_animated_video(agent_history, tracts, video_path, sample_rate=1, fps=5)
        except Exception as e:
            print(f"Could not create video: {e}")
            print("HTML animation created successfully.")
