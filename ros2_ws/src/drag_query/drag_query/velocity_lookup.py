#!/usr/bin/env python3
"""
Utility functions to query the velocity field lookup database.
Velocities are stored as 2D vectors [vx, vy].

Example usage:
  lookup = VelocityLookup('velocity_lookup.h5')
  vx, vy = lookup.get_velocity(timestamp=100, x=507.28, y=206.89)
  coords, vels = lookup.get_all_velocities(timestamp=100)
  speed = lookup.get_speed(timestamp=100, x=507.28, y=206.89)
"""

import h5py
import numpy as np
from scipy.spatial import KDTree

class VelocityLookup:
    """Access velocity field data from HDF5 lookup file."""
    
    def __init__(self, filepath='velocity_lookup.h5'):
        self.filepath = filepath
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from HDF5 file."""
        with h5py.File(self.filepath, 'r') as f:
            self.n_timestamps = f.attrs.get('n_timestamps', 0)
            self.timestamps = sorted([int(k.replace('timestamp_', '')) for k in f.keys()])
    
    def get_all_velocities(self, timestamp):
        """Get all coordinates and velocity vectors for a given timestamp.
        
        Returns:
            coords: (N, 2) array of [x, y] coordinates
            vels: (N, 2) array of [vx, vy] velocity vectors
        """
        key = f'timestamp_{timestamp:04d}'
        with h5py.File(self.filepath, 'r') as f:
            if key not in f:
                raise ValueError(f"Timestamp {timestamp} not found")
            group = f[key]
            coords = group['coordinates'][:]
            vels = group['velocities'][:]
        return coords, vels
    
    def get_velocity_at_point(self, timestamp, x, y, tolerance=1e-3):
        """Get velocity vector at a specific (x, y) coordinate.
        
        Args:
            timestamp: Time index
            x, y: Coordinates
            tolerance: Distance tolerance for matching points
        
        Returns:
            [vx, vy] array or None if no point found within tolerance
        """
        coords, vels = self.get_all_velocities(timestamp)
        
        # Find nearest point
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        if min_dist <= tolerance:
            return vels[min_dist_idx]
        return None
    
    def get_nearest_velocity(self, timestamp, x, y):
        """Get velocity vector at nearest grid point to (x, y).
        
        Returns:
            [vx, vy], [x_nearest, y_nearest]
        """
        coords, vels = self.get_all_velocities(timestamp)
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        min_dist_idx = np.argmin(distances)
        return vels[min_dist_idx], coords[min_dist_idx]
    
    def get_speed(self, timestamp, x, y, tolerance=1e-3):
        """Get speed (magnitude of velocity) at a point."""
        vel = self.get_velocity_at_point(timestamp, x, y, tolerance=tolerance)
        if vel is None:
            return None
        return float(np.sqrt(np.sum(vel**2)))
    
    def get_velocities_in_region(self, timestamp, x_min, x_max, y_min, y_max):
        """Get all velocity vectors within a rectangular region."""
        coords, vels = self.get_all_velocities(timestamp)
        mask = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & \
               (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        return coords[mask], vels[mask]
    
    def get_velocity_time_series(self, x, y, tolerance=1e-2):
        """Get velocity vector evolution at a point across all timestamps.
        
        Returns:
            {timestamp: [vx, vy]}
        """
        time_series = {}
        for timestamp in self.timestamps:
            vel = self.get_velocity_at_point(timestamp, x, y, tolerance=tolerance)
            if vel is not None:
                time_series[timestamp] = vel
        return time_series
    
    def print_info(self):
        """Print database info."""
        print(f"Velocity Lookup Database: {self.filepath}")
        print(f"  Timestamps: {len(self.timestamps)}")
        print(f"  Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        # Sample first timestamp
        with h5py.File(self.filepath, 'r') as f:
            first_key = f'timestamp_{self.timestamps[0]:04d}'
            group = f[first_key]
            n_points = group.attrs['n_points']
            vels = group['velocities'][:]
        print(f"  Points per timestamp: {n_points}")
        print(f"  Velocity components: {vels.shape[1]} (vx, vy)")

if __name__ == '__main__':
    # Example usage
    lookup = VelocityLookup()
    lookup.print_info()
    
    # Query a single point
    print("\nExample queries:")
    vel = lookup.get_velocity_at_point(100, 507.28, 206.89, tolerance=0.1)
    if vel is not None:
        if len(vel) >= 2:
            print(f"  Velocity at (507.28, 206.89) t=100: ({vel[0]:.4f}, {vel[1]:.4f}) m/s")
        else:
            print(f"  Velocity at (507.28, 206.89) t=100: {vel}")
    
    # Nearest point
    vel, nearest_coord = lookup.get_nearest_velocity(100, 507.28, 206.89)
    if len(vel) >= 2:
        print(f"  Nearest to (507.28, 206.89): ({vel[0]:.4f}, {vel[1]:.4f}) m/s at {nearest_coord}")
    else:
        print(f"  Nearest to (507.28, 206.89): {vel} at {nearest_coord}")
    
