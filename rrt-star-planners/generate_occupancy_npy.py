"""
Create a ROS-style occupancy grid map with visualization
- Inside polygon (water) = 0 (free space) = WHITE
- Outside polygon (land) = 100 (occupied) = BLACK
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon, Point
import json
import os

def parse_geo_file(filepath):
    """Parse points and edges from .geo file"""
    points = {}
    edges = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Parse points
        point_match = re.match(r'Point\((\d+)\)\s*=\s*\{\s*([-\d.]+),\s*([-\d.]+)', line)
        if point_match:
            pt_id = int(point_match.group(1))
            x = float(point_match.group(2))
            y = float(point_match.group(3))
            points[pt_id] = (x, y)
        
        # Parse edges
        line_match = re.match(r'Line\((\d+)\)\s*=\s*\{(\d+),\s*(\d+)\}', line)
        if line_match:
            p1 = int(line_match.group(2))
            p2 = int(line_match.group(3))
            edges.append((p1, p2))
    
    return points, edges

def edges_to_polygon_coords(points, edges):
    """Convert edges to ordered polygon coordinates"""
    polygon_coords = []
    current_pt = edges[0][0]
    visited_edges = set()
    
    for _ in range(len(edges)):
        polygon_coords.append(points[current_pt])
        
        # Find next edge
        for edge_idx, (p1, p2) in enumerate(edges):
            if edge_idx not in visited_edges:
                if p1 == current_pt:
                    visited_edges.add(edge_idx)
                    current_pt = p2
                    break
                elif p2 == current_pt:
                    visited_edges.add(edge_idx)
                    current_pt = p1
                    break
    
    return polygon_coords

def create_occupancy_grid(polygon_coords, resolution=5.0):
    """
    Create occupancy grid from polygon
    resolution: grid cell size in coordinate units
    """
    poly_array = np.array(polygon_coords)
    
    # Get bounds with padding
    min_x, min_y = poly_array.min(axis=0)
    max_x, max_y = poly_array.max(axis=0)
    
    padding = 50
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    # Create grid
    grid_width = int(np.ceil((max_x - min_x) / resolution))
    grid_height = int(np.ceil((max_y - min_y) / resolution))
    
    print(f"Grid size: {grid_width} x {grid_height}")
    print(f"Bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
    
    # Create shapely polygon for efficient point-in-polygon test
    polygon = ShapelyPolygon(polygon_coords)
    
    # Initialize grid (100 = occupied/land, 0 = free/water)
    occupancy_grid = np.full((grid_height, grid_width), 100, dtype=np.int8)
    
    # Fill grid
    for i in range(grid_height):
        for j in range(grid_width):
            # Get cell center coordinates
            x = min_x + (j + 0.5) * resolution
            y = min_y + (i + 0.5) * resolution
            
            point = Point(x, y)
            if polygon.contains(point):
                occupancy_grid[i, j] = 0  # Free space (inside)
    
    return occupancy_grid, (min_x, min_y), resolution

# Main
geo_file = '../meshes/sydney_regatta/water_polygon.geo'
points, edges = parse_geo_file(geo_file)
polygon_coords = edges_to_polygon_coords(points, edges)

# Create occupancy grid
grid, origin, resolution = create_occupancy_grid(polygon_coords, resolution=5.0)

map_name = "sydney_regatta_results"

# Save occupancy grid
output_dir = f'lookup_tables/{map_name}'
os.makedirs(output_dir, exist_ok=True)

# Save as NPY (for computation)
np.save(f'{output_dir}/occupancy_grid.npy', grid)

# Save as NPZ (numpy compressed - includes everything easily loadable)
np.savez_compressed(f'{output_dir}/occupancy_grid.npz',
                     grid=grid,
                     origin_x=origin[0],
                     origin_y=origin[1],
                     resolution=resolution)

# Save metadata as JSON (ROS-compatible)
metadata = {
    "origin_x": float(origin[0]),
    "origin_y": float(origin[1]),
    "resolution": float(resolution),
    "width": int(grid.shape[1]),
    "height": int(grid.shape[0]),
    "threshold_occupied": 50,
    "threshold_free": 0
}

with open(f'{output_dir}/occupancy_grid_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Occupancy grid saved: {output_dir}/occupancy_grid.npy")
print(f"✓ Occupancy grid (compressed) saved: {output_dir}/occupancy_grid.npz")
print(f"✓ Metadata saved: {output_dir}/occupancy_grid_metadata.json")
print(f"  Grid shape: {grid.shape}")
print(f"  Origin: ({origin[0]:.1f}, {origin[1]:.1f})")
print(f"  Resolution: {resolution} m/cell")
print(f"  Free cells (0): {np.sum(grid == 0)}")
print(f"  Occupied cells (100): {np.sum(grid == 100)}")

# Visualization with white = free, black = occupied
fig, ax = plt.subplots(figsize=(14, 10))

# Create custom colormap: 0 (free) = white, 100 (occupied) = black
from matplotlib.colors import ListedColormap, Normalize
colors = ['white', 'black']
cmap = ListedColormap(colors)
norm = Normalize(vmin=0, vmax=100)

# Display occupancy grid
im = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower', 
               extent=[origin[0], origin[0] + grid.shape[1] * resolution,
                       origin[1], origin[1] + grid.shape[0] * resolution])

# Overlay polygon boundary in blue (hidden)
# poly_patch = MplPolygon(polygon_coords, fill=False, edgecolor='blue', 
#                         linewidth=2.5, label='Water Boundary')
# ax.add_patch(poly_patch)

# Scatter polygon points in red (hidden)
# poly_array = np.array(polygon_coords)
# ax.scatter(poly_array[:, 0], poly_array[:, 1], c='red', s=40, zorder=5, label='Vertices')

ax.set_xlabel('X Coordinate (m)', fontsize=12)
ax.set_ylabel('Y Coordinate (m)', fontsize=12)
ax.set_title('ROS Occupancy Grid Map\n(White=Free Space/Water, Black=Occupied/Land)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.2, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/occupancy_grid_map.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {output_dir}/occupancy_grid_map.png")

# Add helper function for easy loading
print("\n" + "="*60)
print("EASY LOADING:")
print("="*60)
print("Python code to load the occupancy grid:")
print("-"*60)
print("import numpy as np")
print("data = np.load('occupancy_grid.npz')")
print("grid = data['grid']")
print("origin_x = float(data['origin_x'])")
print("origin_y = float(data['origin_y'])")
print("resolution = float(data['resolution'])")
print("-"*60)

plt.show()
