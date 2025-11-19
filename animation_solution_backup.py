"""
BACKUP: Working Animation Solution - Standalone HTML Approach with Map Theme Switcher

This is a backup of the working create_animated_html_map() function
that uses a completely standalone HTML file approach instead of Folium.

Key principle: Create the Leaflet map directly in JavaScript,
don't try to find or control Folium's map object.

Features:
- Standalone HTML with Leaflet.js
- Animation controls (play/pause, prev/next)
- Map theme switcher in controls area
- 5 map themes: Light CartoDB, OpenStreetMap, Dark CartoDB, Terrain Stamen, Toner Stamen

Date: 2024 (Updated with map theme switcher)
"""

def create_animated_html_map(history: list,
                             tracts: gpd.GeoDataFrame,
                             output_path: Path,
                             sample_rate: int = 2) -> None:
    """
    Create an animated HTML map showing agent movement over time.
    
    Uses a standalone HTML file with Leaflet.js - no Folium dependency.
    
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
    
    # Collect step data for JavaScript animation
    import json
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
    
    # Embed step data as JSON
    step_data_json = json.dumps(step_data)
    
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
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div id="animation-controls">
        <h4 style="margin-top: 0;">Animation Controls</h4>
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
        <p><span style="color:#e74c3c">●</span> Black (B)</p>
        <p><span style="color:#3498db">●</span> White (W)</p>
        <p><span style="color:#2ecc71">●</span> Asian (A)</p>
        <p><span style="color:#f39c12">●</span> Hispanic (H)</p>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script>
        // Configuration
        var centerLat = {center_lat};
        var centerLon = {center_lon};
        var stepData = {step_data_json};
        var numSteps = {num_steps};
        
        // Animation state
        var currentStep = 0;
        var isPlaying = false;
        var intervalId = null;
        var speed = 500;
        
        // Map and layers
        var map = null;
        var markerLayer = null;
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
            
            // Clear existing markers
            markerLayer.clearLayers();
            
            // Get markers for this step
            var markers = stepData[step];
            if (!markers || markers.length === 0) return;
            
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
                var tooltipContent = 'Tract ' + m.tract_id + ': ' + m.total_agents + ' agents';
                
                // Create marker
                var marker = L.circleMarker([m.lat, m.lon], {{
                    radius: m.radius,
                    fillColor: m.color,
                    color: '#333',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7
                }});
                
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
