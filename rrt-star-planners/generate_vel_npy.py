import os
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import meshio

# --- 1. Load VTU files ---
print("--- Loading VTU files ---")
directory_path = '../velocity_fields/sydney_regatta/raw/Velocity2d'
output_folder = './lookup_tables/sydney_regatta'

vtu_files = []
loaded_vtu_data = []
if os.path.exists(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.vtu'):
            vtu_files.append(os.path.join(directory_path, filename))
    print(f"Found {len(vtu_files)} .vtu files:")
    
    for vtu_file in vtu_files:
        try:
            mesh = pv.read(vtu_file)
            loaded_vtu_data.append(mesh)
        except Exception as e:
            print(f"Error loading {vtu_file}: {e}")
    
    print(f"Loaded {len(loaded_vtu_data)} VTU datasets.")
    if loaded_vtu_data:
        print(f"First dataset info:\n{loaded_vtu_data[0]}")
else:
    print(f"Directory not found: {directory_path}")

# --- 2. Matplotlib Plot of First VTU Dataset ---
print("\n--- Plotting First VTU Dataset (Matplotlib) ---")
if loaded_vtu_data:
    mesh = loaded_vtu_data[0]

    # Extract points (coordinates)
    points = mesh.points
    x = points[:, 0]
    y = points[:, 1]

    # Check for scalar arrays to color the plot
    scalar_data_first_mesh = None
    array_name_first_mesh = ''
    if mesh.n_arrays > 0:
        array_name_first_mesh = mesh.array_names[0]
        raw_scalar_data = mesh[array_name_first_mesh]

        scalar_data_first_mesh = np.asarray(raw_scalar_data, dtype=float)
        scalar_data_first_mesh = np.nan_to_num(scalar_data_first_mesh, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

        if scalar_data_first_mesh.ndim > 1 and scalar_data_first_mesh.shape[1] == 1:
            scalar_data_first_mesh = scalar_data_first_mesh.flatten()
        elif scalar_data_first_mesh.ndim > 1:
            print(f"Warning: Scalar data '{array_name_first_mesh}' is not 1D or (N,1). Taking the first column. Original shape: {raw_scalar_data.shape}")
            scalar_data_first_mesh = scalar_data_first_mesh[:, 0]

        print(f"Using scalar array for first mesh plot: {array_name_first_mesh}")
        if scalar_data_first_mesh.size > 0:
            print(f"Scalar data min: {scalar_data_first_mesh.min()}, max: {scalar_data_first_mesh.max()}")

    plt.figure(figsize=(10, 8))
    if scalar_data_first_mesh is not None and scalar_data_first_mesh.size > 0:
        plt.scatter(x, y, c=scalar_data_first_mesh, cmap='viridis', s=1)
        plt.colorbar(label=array_name_first_mesh)
    else:
        plt.scatter(x, y, s=1)

    plt.title('Matplotlib Plot of First VTU Dataset Points')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
else:
    print("No VTU data loaded to plot with Matplotlib.")

# --- 3. Average of All VTU Data ---
print("\n--- Calculating Average of All Scalar Data ---")
if loaded_vtu_data:
    all_scalar_values = []
    array_name_overall = ''
    for i, mesh in enumerate(loaded_vtu_data):
        if mesh.n_arrays > 0:
            array_name_overall = mesh.array_names[0]
            raw_scalar_data = mesh[array_name_overall]

            scalar_data = np.asarray(raw_scalar_data, dtype=float)
            scalar_data = np.nan_to_num(scalar_data, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            if scalar_data.ndim > 1 and scalar_data.shape[1] == 1:
                scalar_data = scalar_data.flatten()
            elif scalar_data.ndim > 1:
                scalar_data = scalar_data[:, 0]

            if scalar_data.size > 0:
                all_scalar_values.append(scalar_data)
        else:
            print(f"Warning: Mesh {i} has no scalar arrays.")

    if all_scalar_values:
        combined_scalar_values = np.concatenate(all_scalar_values)
        average_of_all_vtu_data = np.mean(combined_scalar_values)
        print(f"The average of all scalar data ('{array_name_overall}') across all VTU files is: {average_of_all_vtu_data:.4f}")
    else:
        print("No valid scalar data was collected from the VTU files to calculate an average.")
else:
    print("No VTU data loaded to calculate the average.")

# --- 4. Calculate Average Velocity Field and Magnitude ---
print("\n--- Calculating Average Velocity Field and Magnitude ---")
if loaded_vtu_data:
    if not loaded_vtu_data[0].n_arrays > 0 or 'Depth averaged velocity' not in loaded_vtu_data[0].array_names:
        print("No 'Depth averaged velocity' array found in the first VTU file for average field calculation.")
    else:
        num_points = loaded_vtu_data[0].n_points
        total_velocity_field = np.zeros((num_points, 3))
        num_meshes = len(loaded_vtu_data)

        for mesh in loaded_vtu_data:
            if 'Depth averaged velocity' in mesh.array_names:
                velocity_vectors = mesh['Depth averaged velocity']
                if velocity_vectors.shape == (num_points, 3):
                    total_velocity_field += velocity_vectors
                else:
                    print(f"Warning: Velocity array shape mismatch for average field. Expected ({num_points}, 3), got {velocity_vectors.shape}. Skipping mesh.")
            else:
                print("Warning: 'Depth averaged velocity' not found in a mesh for average field. Skipping.")

        if num_meshes > 0:
            average_velocity_field = total_velocity_field / num_meshes
            average_velocity_magnitude = np.linalg.norm(average_velocity_field, axis=1)
            print(f"Average velocity field and magnitude calculated for {num_points} points.")
        else:
            print("No meshes processed to calculate average velocity field.")
else:
    print("No VTU data loaded to calculate the average velocity field.")

# --- 5. Matplotlib Plot of Average Velocity Magnitude with Boundaries ---
print("\n--- Plotting Average Velocity Magnitude (Matplotlib) ---")
if 'average_velocity_magnitude' in locals() and loaded_vtu_data:
    mesh_template = loaded_vtu_data[0]
    points_avg = mesh_template.points
    x_avg = points_avg[:, 0]
    y_avg = points_avg[:, 1]

    magnitude_to_plot = np.asarray(average_velocity_magnitude, dtype=float)
    magnitude_to_plot = np.nan_to_num(magnitude_to_plot, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    plt.figure(figsize=(10, 8))
    if magnitude_to_plot.size > 0:
        plt.scatter(x_avg, y_avg, c=magnitude_to_plot, cmap='viridis', s=1)
        plt.colorbar(label='Average Velocity Magnitude')
        print(f"Average Velocity Magnitude min: {magnitude_to_plot.min():.4f}, max: {magnitude_to_plot.max():.4f}")
    else:
        plt.scatter(x_avg, y_avg, s=1)
        print("No valid average velocity magnitude data to plot.")
    plt.title('Matplotlib Plot of Average Velocity Magnitude with Mesh Boundary')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
else:
    print("Average velocity magnitude data not found or no VTU data loaded.")

# --- 6. Build and Save Regular Velocity Lookup Tables (u, v, |v|) ---
print("\n--- Building Regular Velocity Lookup Tables ---")
if 'average_velocity_field' in locals() and loaded_vtu_data:
    mesh_template = loaded_vtu_data[0]
    pts = mesh_template.points
    x_pts = pts[:, 0]
    y_pts = pts[:, 1]

    # Average velocity components at mesh points
    u_vals = average_velocity_field[:, 0]
    v_vals = average_velocity_field[:, 1]
    mag_vals = np.sqrt(u_vals**2 + v_vals**2)

    # Regular grid (same style as existing lookup files)
    nx, ny = 300, 300
    x_lin = np.linspace(x_pts.min(), x_pts.max(), nx)
    y_lin = np.linspace(y_pts.min(), y_pts.max(), ny)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin, indexing='ij')

    # Interpolate to regular grid
    points_xy = np.column_stack((x_pts, y_pts))
    grid_u = griddata(points_xy, u_vals, (grid_x, grid_y), method='linear')
    grid_v = griddata(points_xy, v_vals, (grid_x, grid_y), method='linear')
    grid_mag = griddata(points_xy, mag_vals, (grid_x, grid_y), method='linear')

    # Fill NaNs near/outside convex hull with nearest
    if np.isnan(grid_u).any():
        grid_u_nearest = griddata(points_xy, u_vals, (grid_x, grid_y), method='nearest')
        grid_u = np.where(np.isnan(grid_u), grid_u_nearest, grid_u)
    if np.isnan(grid_v).any():
        grid_v_nearest = griddata(points_xy, v_vals, (grid_x, grid_y), method='nearest')
        grid_v = np.where(np.isnan(grid_v), grid_v_nearest, grid_v)
    if np.isnan(grid_mag).any():
        grid_mag_nearest = griddata(points_xy, mag_vals, (grid_x, grid_y), method='nearest')
        grid_mag = np.where(np.isnan(grid_mag), grid_mag_nearest, grid_mag)

    os.makedirs(output_folder, exist_ok=True)
    np.save(f'{output_folder}/grid_x.npy', grid_x)
    np.save(f'{output_folder}/grid_y.npy', grid_y)
    np.save(f'{output_folder}/average_velocity_u_lookup.npy', grid_u)
    np.save(f'{output_folder}/average_velocity_v_lookup.npy', grid_v)
    np.save(f'{output_folder}/average_velocity_magnitude_lookup.npy', grid_mag)

    print(f"Saved lookup tables to: {output_folder}")
    print("  - grid_x.npy")
    print("  - grid_y.npy")
    print("  - average_velocity_u_lookup.npy")
    print("  - average_velocity_v_lookup.npy")
    print("  - average_velocity_magnitude_lookup.npy")
else:
    print("Average velocity field not available; skipping lookup-table generation.")

# --- 7. Identify and Visualize Outer Boundary Points ---
print("\n--- Identifying Outer Boundary Points ---")
if loaded_vtu_data:
    from collections import defaultdict
    mesh_template = loaded_vtu_data[0]
    
    # Find all edges and count how many cells they belong to
    edge_to_cells = defaultdict(set)
    
    for cell_id in range(mesh_template.n_cells):
        cell = mesh_template.get_cell(cell_id)
        cell_points = cell.points
        n_pts = len(cell_points)
        
        # Iterate through each edge of the cell
        for i in range(n_pts):
            pt1 = cell_points[i]
            pt2 = cell_points[(i + 1) % n_pts]
            
            # Create a canonical edge key (sorted by coordinates)
            edge_key = tuple(sorted([tuple(pt1[:2]), tuple(pt2[:2])]))
            edge_to_cells[edge_key].add(cell_id)
    
    # Find boundary edges (edges in only 1 cell) and their points
    boundary_point_set = set()
    boundary_edges = []
    
    for edge_key, cell_ids in edge_to_cells.items():
        if len(cell_ids) == 1:  # Boundary edge
            boundary_point_set.add(edge_key[0])
            boundary_point_set.add(edge_key[1])
            boundary_edges.append(edge_key)
    
    # Convert boundary points to arrays
    boundary_pts = np.array(list(boundary_point_set))
    boundary_x = boundary_pts[:, 0]
    boundary_y = boundary_pts[:, 1]
    
    # Get inner points (not on boundary)
    all_points = mesh_template.points
    all_x = all_points[:, 0]
    all_y = all_points[:, 1]
    
    inner_point_set = set()
    for pt in all_points:
        pt_key = tuple(pt[:2])
        if pt_key not in boundary_point_set:
            inner_point_set.add(pt_key)
    
    inner_pts = np.array(list(inner_point_set)) if inner_point_set else np.empty((0, 2))
    
    print(f"Total points: {len(all_points)}")
    print(f"Boundary points (outer): {len(boundary_pts)}")
    print(f"Interior points: {len(inner_pts)}")
    print(f"Boundary edges: {len(boundary_edges)}")
    
    # Visualize outer points vs inner points
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: All points with boundary highlighted
    ax = axes[0]
    ax.scatter(all_x, all_y, c='lightblue', s=2, alpha=0.3, label='Interior points')
    ax.scatter(boundary_x, boundary_y, c='red', s=20, label='Boundary points', zorder=5)
    ax.set_title('Mesh: Interior vs Boundary Points')
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Boundary outline only
    ax = axes[1]
    ax.scatter(boundary_x, boundary_y, c='red', s=15)
    # Draw boundary edges as lines
    for edge in boundary_edges[:200]:  # Draw first 200 edges for clarity
        pt1, pt2 = edge
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=1.5, alpha=0.6)
    
    ax.set_title('Boundary Outline (Coastline)')
    ax.set_xlabel('X-coordinate (m)')
    ax.set_ylabel('Y-coordinate (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save boundary points to file
    boundary_data = np.column_stack([boundary_x, boundary_y])
    np.save(f'{output_folder}/boundary_points.npy', boundary_data)
    print(f"\nBoundary points saved to {output_folder}/boundary_points.npy")
    
else:
    print("No VTU data loaded to identify boundary points.")



