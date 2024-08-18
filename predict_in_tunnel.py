import os
import sys
import math
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import KDTree

# Atom types for binary tagging
atom_types = {
    'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'OXT': 4, 'CB': 5, 'CG': 6, 'CD': 7, 'CE': 8, 'CZ': 9, 
    'OG': 10, 'SG': 11, 'Others': 12
}

# Define the amino acids and other residue types for binary tagging
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
other_residues = ['CU', 'MG', 'SE', 'HEA', 'TRD', 'DMU', 'Others']
backbone_atoms = ['N', 'CA', 'C', 'O', 'OXT']
residue_types = {aa: i for i, aa in enumerate(amino_acids)}
residue_types.update({res: len(amino_acids) + i for i, res in enumerate(other_residues)})
backbone_residue_index = len(amino_acids) + len(other_residues)

def parse_pdb_file(file_path):
    print(f"Parsing PDB file {file_path}...")
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

def parse_sphere_pdb_file(file_path):
    print(f"Parsing sphere PDB file {file_path}...")
    spheres = []
    with open(file_path, 'r', encoding='utf-8') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                radius_str = line[60:66].strip()
                if radius_str:  # Ensure the radius field is not empty
                    try:
                        radius = float(radius_str)  # Use B-factor as radius
                        spheres.append((np.array([x, y, z]), radius))
                    except ValueError:
                        print(f"Skipping line: invalid radius value '{radius_str}' in file {file_path}.")
    return spheres

def is_within_tunnel(point, spheres):
    for center, radius in spheres:
        if np.linalg.norm(point - center) <= radius:
            return True
    return False

def create_grid(min_coord, max_coord, step=1.0, kd_tree=None, distance_threshold=3.0, spheres=None):
    print("Creating grid points within the defined box...")
    grid_points = []

    for z in np.arange(min_coord[2], max_coord[2] + step, step):
        x_range = np.arange(min_coord[0], max_coord[0] + step, step)
        y_range = np.arange(min_coord[1], max_coord[1] + step, step)
        xy_plane = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
        for xy in xy_plane:
            point = np.array([xy[0], xy[1], z])
            if spheres and not is_within_tunnel(point, spheres):
                continue
            if kd_tree is not None:
                distance, _ = kd_tree.query(point)
                if distance <= distance_threshold:
                    grid_points.append(point)
            else:
                grid_points.append(point)

    grid_points = np.array(grid_points)
    print(f"Total grid points after filtering: {len(grid_points)}")
    return grid_points

def find_closest_atoms(grid_point, atoms, num_closest=10):
    distances = [(calculate_distance({'x_coordinate': grid_point[0], 'y_coordinate': grid_point[1], 'z_coordinate': grid_point[2]}, atom), atom) for atom in atoms if atom['element_symbol'] != 'H']
    distances.sort(key=lambda x: x[0])
    closest_atoms = distances[:num_closest]
    return closest_atoms

def calculate_distance(atom1, atom2):
    return math.sqrt(
        (atom1['x_coordinate'] - atom2['x_coordinate']) ** 2 +
        (atom1['y_coordinate'] - atom2['y_coordinate']) ** 2 +
        (atom2['z_coordinate'] - atom1['z_coordinate']) ** 2
    )

def calculate_vector(atom1, atom2):
    return (
        atom2['x_coordinate'] - atom1['x_coordinate'],
        atom2['y_coordinate'] - atom1['y_coordinate'],
        atom2['z_coordinate'] - atom1['z_coordinate']
    )

def dot_product(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec2[2]*vec2[2]

def vector_magnitude(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def calculate_cos_theta(vec1, vec2):
    mag1 = vector_magnitude(vec1)
    mag2 = vector_magnitude(vec2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product(vec1, vec2) / (mag1 * mag2)

def calculate_cos_phi(vec1, vec2, vec3):
    normal1 = np.cross(vec1, vec2)
    normal2 = np.cross(vec2, vec3)
    mag1 = vector_magnitude(normal1)
    mag2 = vector_magnitude(normal2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product(normal1, normal2) / (mag1 * mag2)

def get_binary_tags(element_symbol):
    tags = [0] * len(atom_types)
    if element_symbol in atom_types:
        tags[atom_types[element_symbol]] = 1
    else:
        tags[atom_types['Others']] = 1
    return tags

def get_residue_tags(residue_name, atom_name):
    tags = [0] * (len(amino_acids) + len(other_residues) + 1)
    if atom_name in backbone_atoms:
        tags[backbone_residue_index] = 1
    elif residue_name in residue_types:
        tags[residue_types[residue_name]] = 1
    else:
        tags[residue_types['Others']] = 1
    return tags

def prepare_features(closest_atoms):
    features = []
    target_atom = closest_atoms[0][1]
    for i, (distance, atom) in enumerate(closest_atoms):
        element = atom['element_symbol']
        residue = atom['residue_name']
        atom_name = atom['element_symbol']
        binary_tags = get_binary_tags(element)
        residue_tags = get_residue_tags(residue, atom_name)
        xr = atom['x_coordinate'] - target_atom['x_coordinate']
        yr = atom['y_coordinate'] - target_atom['y_coordinate']
        zr = atom['z_coordinate'] - target_atom['z_coordinate']
        r = distance
        cos_theta = cos_phi = 0
        
        if i > 0:
            prev_atom = closest_atoms[i-1][1]
            vec1 = calculate_vector(target_atom, prev_atom)
            vec2 = calculate_vector(target_atom, atom)
            cos_theta = calculate_cos_theta(vec1, vec2)
        
        if i > 1:
            prev_prev_atom = closest_atoms[i-2][1]
            vec3 = calculate_vector(target_atom, prev_prev_atom)
            cos_phi = calculate_cos_phi(vec1, vec2, vec3)
        
        features.extend(binary_tags + residue_tags + [xr, yr, zr, r, cos_theta, cos_phi])
    
    if len(features) < 470:
        features.extend([0] * (470 - len(features)))
    else:
        features = features[:470]
    
    return features

def make_predictions(model, features):
    features = np.array(features).reshape(1, -1)
    predictions = model.predict(features)
    return predictions

def load_trained_model(model_path):
    return load_model(model_path)

def save_predictions_to_pdb(predictions, coords, output_path, filter_threshold=None):
    print(f"Saving predictions to PDB file at {output_path}...")
    with open(output_path, 'w') as f:
        for i, (coord, prob) in enumerate(zip(coords, predictions)):
            if filter_threshold is None or prob >= filter_threshold:
                x, y, z = coord
                x, y, z, prob = x.item(), y.item(), z.item(), prob.item()
                f.write(f"ATOM  {i+1:5d}  O   HOH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {prob:6.4f}           O\n")
    print(f"Predictions saved to {output_path}")

def cluster_predictions(coords, predictions, min_coord, max_coord, grid_size):
    print("Clustering predictions...")
    coords = np.array(coords)
    predictions = np.array(predictions)
    
    x_bins = np.arange(min_coord[0], max_coord[0] + grid_size, grid_size)
    y_bins = np.arange(min_coord[1], max_coord[1] + grid_size, grid_size)
    z_bins = np.arange(min_coord[2], max_coord[2] + grid_size, grid_size)

    clustered_coords = []
    clustered_predictions = []

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            for k in range(len(z_bins) - 1):
                mask = (
                    (coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i + 1]) &
                    (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j + 1]) &
                    (coords[:, 2] >= z_bins[k]) & (coords[:, 2] < z_bins[k + 1])
                )
                if np.any(mask):
                    max_pred_idx = np.argmax(predictions[mask])
                    clustered_coords.append(coords[mask][max_pred_idx])
                    clustered_predictions.append(predictions[mask][max_pred_idx])
    
    return clustered_coords, clustered_predictions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_prediction.py <background_pdb_file> <sphere_pdb_directory>")
        sys.exit(1)

    background_pdb_file = sys.argv[1]  # Single PDB file used as the background
    sphere_pdb_directory = sys.argv[2]  # Directory containing the PDB files defining spherical regions

    model_path = os.path.join('trained_modules', 'water_model.keras')

    # Load the trained model
    model = load_trained_model(model_path)
    print("Loaded trained model.")

    # Parse the background PDB file to get coordinates
    reference_points, atoms = parse_pdb_file(background_pdb_file)
    print(f"Parsed {len(reference_points)} coordinates from the background PDB file.")

    # Determine the range of coordinates for the background PDB file
    min_coord = reference_points.min(axis=0)
    max_coord = reference_points.max(axis=0)
    print(f"Background coordinate range: Min {min_coord}, Max {max_coord}")

    # Build a KDTree for the background protein atoms for efficient distance queries
    kd_tree = KDTree(reference_points)
    distance_threshold = 3.0

    # Process each sphere PDB file in the sphere directory
    for sphere_file_name in os.listdir(sphere_pdb_directory):
        sphere_file_path = os.path.join(sphere_pdb_directory, sphere_file_name)
        
        if not os.path.isfile(sphere_file_path):
            continue

        output_folder = os.path.join(sphere_pdb_directory, 'predictions', sphere_file_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pdb_output_path_total = os.path.join(output_folder, f'{sphere_file_name}_total.pdb')
        pdb_output_path_clustered = os.path.join(output_folder, f'{sphere_file_name}_clustered.pdb')

        # Parse the sphere PDB file to get the spherical regions (for tunnel boundaries)
        spheres = parse_sphere_pdb_file(sphere_file_path)
        print(f"Parsed {len(spheres)} spheres from {sphere_file_name}.")

        # Create a grid within the defined box, and filter points within distance threshold and spheres
        filtered_grid_points = create_grid(min_coord, max_coord, step=1.0, kd_tree=kd_tree, distance_threshold=distance_threshold, spheres=spheres)
        print(f"Generated {len(filtered_grid_points)} filtered grid points for {sphere_file_name}.")

        if len(filtered_grid_points) == 0:
            print(f"No valid grid points found for {sphere_file_name}, skipping this file.")
            continue

        all_predictions = []

        # Generate predictions for each grid point
        num_filtered_grid_points = len(filtered_grid_points)
        for i, grid_point in enumerate(filtered_grid_points):
            print(f"Processing filtered grid point {i+1}/{num_filtered_grid_points} for {sphere_file_name}...")
            closest_atoms = find_closest_atoms(grid_point, atoms, num_closest=10)
            features = prepare_features(closest_atoms)
            prediction = make_predictions(model, features)
            all_predictions.append((grid_point[0], grid_point[1], grid_point[2], prediction[0][0]))

        if len(all_predictions) == 0:
            print(f"No predictions generated for {sphere_file_name}, skipping this file.")
            continue

        all_predictions = sorted(all_predictions, key=lambda x: -x[3])

        coords, probs = zip(*[(pred[:3], pred[3]) for pred in all_predictions])

        # Save all predictions to PDB file
        save_predictions_to_pdb(probs, coords, pdb_output_path_total)

        # Cluster predictions and save to PDB file
        grid_size = 3.0
        clustered_coords, clustered_probs = cluster_predictions(coords, probs, min_coord, max_coord, grid_size)
        save_predictions_to_pdb(clustered_probs, clustered_coords, pdb_output_path_clustered)

    print("Prediction process completed for all files.")
