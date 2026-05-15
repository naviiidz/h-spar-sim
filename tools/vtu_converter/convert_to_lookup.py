#!/usr/bin/env python3
"""
Convert all VTU files to HDF5 lookup database.
Creates efficient lookup: /timestamp_XXX/coordinates and /timestamp_XXX/velocities
"""

import os
import sys
import numpy as np
import glob
import struct
from xml.etree import ElementTree as ET

def extract_xml_header(filepath):
    """Extract XML header from VTU file with appended binary data."""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    appended_marker = b'<AppendedData encoding="raw">'
    idx = content.find(appended_marker)
    
    if idx == -1:
        return content.decode('utf-8', errors='ignore')
    
    xml_part = content[:idx].decode('utf-8', errors='ignore')
    return xml_part + '</VTKFile>'

def read_binary_data(filepath, offset, dtype, count):
    """Read binary data from appended section of VTU file."""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    appended_marker = b'<AppendedData encoding="raw">\n_'
    idx = content.find(appended_marker)
    
    if idx == -1:
        return None
    
    binary_start = idx + len(appended_marker)
    data_offset = binary_start + offset
    
    if dtype == 'Float64':
        fmt = '<d'
    elif dtype == 'Float32':
        fmt = '<f'
    elif dtype == 'Int32':
        fmt = '<i'
    elif dtype == 'UInt8':
        fmt = 'B'
    else:
        return None
    
    size_bytes = content[data_offset:data_offset+4]
    if len(size_bytes) < 4:
        return None
    size = struct.unpack('<I', size_bytes)[0]
    
    data_start = data_offset + 4
    data_bytes = content[data_start:data_start + size]
    
    if len(data_bytes) < size:
        return None
        
    values = np.frombuffer(data_bytes, dtype=np.dtype(fmt), count=count)
    
    return values

def extract_timestamp(filepath):
    """Extract timestamp from filename (e.g., Velocity2d_123.vtu -> 123)."""
    basename = os.path.basename(filepath)
    # Remove extension and prefix
    parts = basename.replace('.vtu', '').split('_')
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None

def read_vtu_data(filepath):
    """Read coordinates and velocity from a single VTU file."""
    try:
        xml_str = extract_xml_header(filepath)
        root = ET.fromstring(xml_str)
        
        # Get number of points
        piece = root.find('.//Piece')
        if piece is None:
            return None
        
        n_points = int(piece.get('NumberOfPoints', 0))
        
        # Find coordinates
        coords_data = None
        velocity_data = None
        
        for data_array in root.findall('.//DataArray'):
            name = data_array.get('Name')
            if name in ('firedrake_default_coordinates', 'Coordinates'):
                coords_data = data_array
            elif name in ('Velocity2d', 'Depth averaged velocity', 'velocity'):
                velocity_data = data_array
        
        if coords_data is None or velocity_data is None:
            return None
        
        # Read coordinates
        coords_dtype = coords_data.get('type')
        coords_offset = int(coords_data.get('offset', 0))
        coords_n_comp = int(coords_data.get('NumberOfComponents', 3))
        coords = read_binary_data(filepath, coords_offset, coords_dtype, n_points * coords_n_comp)
        
        if coords is None:
            return None
        
        coords = coords.reshape(-1, coords_n_comp)
        
        # Read velocity
        vel_dtype = velocity_data.get('type')
        vel_offset = int(velocity_data.get('offset', 0))
        vel_n_comp = int(velocity_data.get('NumberOfComponents', 2))
        velocity = read_binary_data(filepath, vel_offset, vel_dtype, n_points * vel_n_comp)
        
        if velocity is None:
            return None
        
        velocity = velocity.reshape(-1, vel_n_comp)
        
        return coords, velocity
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None

def main():
    import h5py
    
    # Get search path
    if len(sys.argv) > 1:
        search_path = sys.argv[1]
    else:
        search_path = "../sydney_regatta/raw/Velocity2d/*.vtu"
    
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No VTU files found at {search_path}")
        return
    
    print(f"Found {len(files)} VTU files")
    
    # Create HDF5 file for efficient storage
    output_h5 = 'velocity_lookup.h5'
    
    with h5py.File(output_h5, 'w') as f:
        # Store metadata
        f.attrs['n_timestamps'] = len(files)
        f.attrs['description'] = 'Velocity field lookup: timestamp -> (x, y) -> velocity magnitude'
        
        # Process each file and write directly to HDF5
        for i, filepath in enumerate(files):
            timestamp = extract_timestamp(filepath)
            if timestamp is None:
                print(f"Skipping {filepath} - could not extract timestamp")
                continue
            
            print(f"[{i+1}/{len(files)}] Processing timestamp {timestamp:04d}...", end=' ', flush=True)
            
            result = read_vtu_data(filepath)
            if result is None:
                print("FAILED")
                continue
            
            coords, velocity = result
            n_points = len(velocity)
            
            # Keep only 2D components (vx, vy)
            velocity_2d = velocity[:, :2]
            
            # Write to HDF5
            group = f.create_group(f'timestamp_{timestamp:04d}')
            group.create_dataset('coordinates', data=coords[:, :2], compression='gzip', compression_opts=4)
            group.create_dataset('velocities', data=velocity_2d, compression='gzip', compression_opts=4)
            group.attrs['n_points'] = n_points
            group.attrs['timestamp'] = timestamp
            
            print(f"OK ({n_points} points)")
    
    print(f"\nSuccessfully saved to {output_h5}")
    
    # Print file size
    file_size_mb = os.path.getsize(output_h5) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Print sample data
    print("\nSample data:")
    with h5py.File(output_h5, 'r') as f:
        keys = sorted(f.keys())
        if keys:
            first_key = keys[0]
            group = f[first_key]
            print(f"  {first_key}: {group.attrs['n_points']} points")
            coords = group['coordinates'][:]
            vels = group['velocities'][:]
            for j in range(min(3, len(coords))):
                vx, vy = vels[j]
                print(f"    ({coords[j, 0]:.2f}, {coords[j, 1]:.2f}) -> ({vx:.4f}, {vy:.4f}) m/s")

if __name__ == "__main__":
    main()
