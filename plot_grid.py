import os
import sys
import math
import numpy as np

# Updated atomic radii for common elements (in Å)
atomic_radii = {
    'H': 0,           # Hydrogen (2.08 Å, but set to 0 due to molecular interactions)
    'C': 0.91,        # Carbon
    'N': 0.92,        # Nitrogen
    'O': 0.65,        # Oxygen
    'S': 1.27,        # Sulfur
    'Se': 1.4         # Selenium
}

def parse_pdb_file(file_path):
    print("Parsing PDB file...")
    coordinates = []
    atoms = []
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if (line.startswith("ATOM") or line.startswith("HETATM")) and (line[21] == 'A' or line[21] == 'B'):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coordinates.append([x, y, z])
                atoms.append({
                    'element_symbol': line[76:78].strip(),
                    'x_coordinate': x,
                    'y_coordinate': y,
                    'z_coordinate': z,
                    'residue_name': line[17:20].strip()
                })
    return np.array(coordinates), atoms

def create_grid(min_coord, max_coord, atoms, step=1.0, exclusion_distance=0.8):
    print("Creating grid points within the defined box...")

    grid_points = []

    for z in np.arange(min_coord[2], max_coord[2] + step, step):
        x_range = np.arange(min_coord[0], max_coord[0] + step, step)
        y_range = np.arange(min_coord[1], max_coord[1] + step, step)
        xy_plane = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
        for xy in xy_plane:
            grid_points.append([xy[0], xy[1], z])

    grid_points = np.array(grid_points)

    print(f"Total grid points before filtering: {len(grid_points)}")

    # Convert atoms to a numpy array for faster computation
    atom_coords = np.array([[atom['x_coordinate'], atom['y_coordinate'], atom['z_coordinate']] for atom in atoms if atom['element_symbol'] != 'H'])

    # Filter out grid points that are too close to any atom except hydrogen
    filtered_grid_points = []
    for i, point in enumerate(grid_points):
        if i % 1000 == 0:
            print(f"Processing grid point {i+1}/{len(grid_points)}...")

        distances = np.linalg.norm(atom_coords - point, axis=1)
        if np.all(distances >= exclusion_distance):
            filtered_grid_points.append(point)

    return np.array(filtered_grid_points), grid_points

def save_grid_as_pdb(grid_points, output_file, atom_type='H'):
    print(f"Saving {atom_type} grid points as PDB file...")
    with open(output_file, 'w') as pdb_file:
        for i, point in enumerate(grid_points):
            x, y, z = point
            pdb_file.write(
                f"ATOM  {i+1:5d}  {atom_type:<2}   {atom_type}   A   {i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_type:<2}\n"
            )
    print(f"{atom_type} grid points saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_grid_creation.py <pdb_file_path>")
        sys.exit(1)

    pdb_file_path = sys.argv[1]  # Path to the PDB file

    # Parse PDB file to get coordinates and atoms
    reference_points, atoms = parse_pdb_file(pdb_file_path)
    print(f"Parsed {len(reference_points)} coordinates from PDB file.")

    # Determine the range of coordinates
    min_coord = reference_points.min(axis=0)
    max_coord = reference_points.max(axis=0)
    print(f"Coordinate range: Min {min_coord}, Max {max_coord}")

    # Create a grid within the defined box
    filtered_grid_points, unfiltered_grid_points = create_grid(min_coord, max_coord, atoms, step=1.0, exclusion_distance=0.8)
    print(f"Generated {len(filtered_grid_points)} filtered grid points.")
    print(f"Generated {len(unfiltered_grid_points)} unfiltered grid points.")

    # Save grid points as PDB files
    save_grid_as_pdb(filtered_grid_points, 'filtered_grid_points.pdb')
    save_grid_as_pdb(unfiltered_grid_points, 'unfiltered_grid_points.pdb', atom_type='O')
 