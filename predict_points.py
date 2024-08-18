import os
import sys
import math
import numpy as np
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model
from scipy.spatial import KDTree

# Define the atom types for binary tagging
atom_types = {'H': 0, 'C': 1, 'O': 2, 'N': 3, 'S': 4, 'Others': 5}

# Define the amino acids and other residue types for binary tagging
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'OTHERS']
other_residues = ['CU', 'MG', 'SE', 'HEA', 'TRD', 'DMU', 'Others']
residue_types = {aa: i for i, aa in enumerate(amino_acids)}
residue_types.update({res: len(amino_acids) + i for i, res in enumerate(other_residues)})

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

def find_closest_atoms(grid_point, atoms, num_closest=10):
    distances = [(calculate_distance({'x_coordinate': grid_point[0], 'y_coordinate': grid_point[1], 'z_coordinate': grid_point[2]}, atom), atom) for atom in atoms if atom['element_symbol'] != 'H']
    distances.sort(key=lambda x: x[0])
    closest_atoms = distances[:num_closest]
    return closest_atoms

def get_binary_tags(element_symbol):
    tags = [0] * 6  # Initialize a list with six zeros
    if element_symbol in atom_types:
        tags[atom_types[element_symbol]] = 1
    else:
        tags[atom_types['Others']] = 1
    return tags

def get_residue_tags(residue_name):
    tags = [0] * (len(amino_acids) + len(other_residues))  # Initialize a list with the correct number of zeros
    if residue_name in residue_types:
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
        binary_tags = get_binary_tags(element)
        residue_tags = get_residue_tags(residue)
        xr = atom['x_coordinate'] - target_atom['x_coordinate']
        yr = atom['y_coordinate'] - target_atom['y_coordinate']
        zr = atom['z_coordinate'] - target_atom['z_coordinate']
        r = distance
        cos_theta = cos_phi = 0  # Default values
        
        if i > 0:  # Calculate cos_theta if there is a previous atom
            prev_atom = closest_atoms[i-1][1]
            vec1 = calculate_vector(target_atom, prev_atom)
            vec2 = calculate_vector(target_atom, atom)
            cos_theta = calculate_cos_theta(vec1, vec2)
        
        if i > 1:  # Calculate cos_phi if there are two previous atoms
            prev_prev_atom = closest_atoms[i-2][1]
            vec3 = calculate_vector(target_atom, prev_prev_atom)
            cos_phi = calculate_cos_phi(vec1, vec2, vec3)
        
        features.extend(binary_tags + residue_tags + [xr, yr, zr, r, cos_theta, cos_phi])
    return features[:400] if len(features) >= 400 else features + [0] * (400 - len(features))

def make_predictions(model, features):
    features = np.array(features).reshape(1, -1)  # Reshape for the model
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
                x, y, z, prob = x.item(), y.item(), z.item(), prob.item()  # Convert arrays to scalars
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
        print("Usage: python predict_points.py <pdb_file_path> <points_pdb_file_path>")
        sys.exit(1)

    pdb_file_path = sys.argv[1]  # Path to the PDB file
    points_pdb_file_path = sys.argv[2]  # Path to the PDB file with points to predict
    model_path = os.path.join('trained_modules', 'water_model.keras')  # Path to the trained model
    output_folder = 'predictions'
    pdb_output_path_all = os.path.join(output_folder, 'predictions_all.pdb')
    pdb_output_path_confident = os.path.join(output_folder, 'predictions_confident.pdb')
    pdb_output_path_clustered = os.path.join(output_folder, 'predictions_clustered.pdb')

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parse PDB file to get coordinates
    reference_points, atoms = parse_pdb_file(pdb_file_path)
    print(f"Parsed {len(reference_points)} coordinates from PDB file.")

    # Parse the points PDB file
    prediction_points, _ = parse_pdb_file(points_pdb_file_path)
    print(f"Parsed {len(prediction_points)} prediction points from PDB file.")

    # Load the trained model
    model = load_trained_model(model_path)
    print("Loaded trained model.")

    all_predictions = []    

    # Generate predictions for each point
    num_prediction_points = len(prediction_points)
    for i, grid_point in enumerate(prediction_points):
        print(f"Processing prediction point {i+1}/{num_prediction_points}...")
        closest_atoms = find_closest_atoms(grid_point, atoms, num_closest=10)
        features = prepare_features(closest_atoms)
        prediction = make_predictions(model, features)
        all_predictions.append((grid_point[0], grid_point[1], grid_point[2], prediction[0][0]))

    all_predictions = sorted(all_predictions, key=lambda x: -x[3])  # Sort by probability

    coords, probs = zip(*[(pred[:3], pred[3]) for pred in all_predictions])

    # Save all predictions to PDB file
    save_predictions_to_pdb(probs, coords, pdb_output_path_all)

    # Save confident predictions to PDB files with varying thresholds
    confidence_thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    for threshold in confidence_thresholds:
        output_path = os.path.join(output_folder, f'predictions_confidence_{threshold}.pdb')
        save_predictions_to_pdb(probs, coords, output_path, filter_threshold=threshold)

    # Cluster predictions and save to PDB file
    min_coord = prediction_points.min(axis=0)
    max_coord = prediction_points.max(axis=0)
    grid_size = 3.0  # 3x3 unit grid for clustering
    clustered_coords, clustered_probs = cluster_predictions(coords, probs, min_coord, max_coord, grid_size)
    save_predictions_to_pdb(clustered_probs, clustered_coords, pdb_output_path_clustered)

