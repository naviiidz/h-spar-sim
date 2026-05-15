# VTU to HDF5 Lookup Converter

A tool to convert VTU (VTK Unstructured Grid) velocity field files into an efficient HDF5 lookup database for fast queries.

## Purpose

This script processes velocity field data stored in VTU format and creates a compressed HDF5 database organized by timestamp. This enables efficient spatial-temporal lookups for applications like drag force calculations.

## What It Does

1. **Reads VTU files** - Extracts XML headers and binary velocity/coordinate data
2. **Organizes by timestamp** - Groups data into timestamped datasets
3. **Compresses efficiently** - Uses gzip compression to reduce file size
4. **Creates lookup structure**:
   ```
   velocity_lookup.h5
   ├── timestamp_0000/
   │   ├── coordinates (N×2 array)
   │   └── velocities (N×2 array)
   ├── timestamp_0001/
   │   ├── coordinates
   │   └── velocities
   └── ...
   ```

## Usage

```bash
# Convert all VTU files from default location
python3 convert_to_lookup.py

# Convert from custom directory
python3 convert_to_lookup.py "/path/to/velocity_files/*.vtu"
```

## Requirements

- Python 3.x
- NumPy
- h5py
- ElementTree (included in Python standard library)

## Output

Creates `velocity_lookup.h5` with:
- Compressed coordinate and velocity datasets
- Metadata (number of timestamps, points per timestamp)
- Sample data printout for verification

## Example

```bash
$ python3 convert_to_lookup.py ../sydney_regatta/raw/Velocity2d/*.vtu
Found 500 VTU files
[1/500] Processing timestamp 0000... OK (1024 points)
[2/500] Processing timestamp 0001... OK (1024 points)
...
Successfully saved to velocity_lookup.h5
File size: 125.3 MB
```

## Input File Format

VTU files should contain:
- `Coordinates` or `firedrake_default_coordinates` data array
- `Velocity2d`, `Depth averaged velocity`, or `velocity` data array
- Filename format: `*_XXXX.vtu` where XXXX is the timestamp
