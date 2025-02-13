import os
import sys
import math
import numpy as np
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model

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

def create_grid(min_coord, max_coord, step=1.0, exclusion_distance=0.8):
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
        if i % 10000 == 0:
            print(f"Processing grid point {i+1}/{len(grid_points)}...")

        distances = np.linalg.norm(atom_coords - point, axis=1)
        if np.all(distances >= exclusion_distance):
            filtered_grid_points.append(point)
            
    return np.array(filtered_grid_points), grid_points

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
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def vector_magnitude(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def calculate_cos_theta(vec1, vec2):
    return dot_product(vec1, vec2) / (vector_magnitude(vec1) * vector_magnitude(vec2))

def calculate_cos_phi(vec1, vec2, vec3):
    normal1 = np.cross(vec1, vec2)
    normal2 = np.cross(vec2, vec3)
    return dot_product(normal1, normal2) / (vector_magnitude(normal1) * vector_magnitude(normal2))

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
    for distance, atom in closest_atoms:
        element = atom['element_symbol']
        residue = atom['residue_name']
        binary_tags = get_binary_tags(element)
        residue_tags = get_residue_tags(residue)
        xr = atom['x_coordinate'] - target_atom['x_coordinate']
        yr = atom['y_coordinate'] - target_atom['y_coordinate']
        zr = atom['z_coordinate'] - target_atom['z_coordinate']
        r = distance
        cos_theta = cos_phi = 0  # Placeholder values
        if len(features) > 0:
            prev_atom = features[-1]
            vec1 = calculate_vector(target_atom, prev_atom)
            vec2 = calculate_vector(target_atom, atom)
            cos_theta = calculate_cos_theta(vec1, vec2)
            if len(features) > 1:
                prev_prev_atom = features[-2]
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

def cluster_predictions(coords, predictions, grid_size):
    print("Clustering predictions...")
    coords = np.array(coords)
    predictions = np.array(predictions)
    db = DBSCAN(eps=grid_size, min_samples=1).fit(coords)
    labels = db.labels_
    
    clustered_coords = []
    clustered_predictions = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        label_mask = (labels == label)
        if np.any(label_mask):
            label_coords = coords[label_mask]
            label_predictions = predictions[label_mask]
            max_pred_idx = np.argmax(label_predictions)
            clustered_coords.append(label_coords[max_pred_idx])
            clustered_predictions.append(label_predictions[max_pred_idx])
    
    return clustered_coords, clustered_predictions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_prediction.py <pdb_file_path>")
        sys.exit(1)

    pdb_file_path = sys.argv[1]  # Path to the PDB file
    model_path = os.path.join('trained_modules', 'water_model.keras')  # Path to the trained model
    output_folder = 'predictions'
    reference_points_output_path = os.path.join(output_folder, 'reference_points.txt')
    pdb_output_path_all = os.path.join(output_folder, 'predictions_all.pdb')
    pdb_output_path_confident = os.path.join(output_folder, 'predictions_confident.pdb')
    pdb_output_path_clustered = os.path.join(output_folder, 'predictions_clustered.pdb')

    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parse PDB file to get coordinates
    reference_points, atoms = parse_pdb_file(pdb_file_path)
    print(f"Parsed {len(reference_points)} coordinates from PDB file.")

    # Determine the range of coordinates
    min_coord = reference_points.min(axis=0)
    max_coord = reference_points.max(axis=0)
    print(f"Coordinate range: Min {min_coord}, Max {max_coord}")

    # Save reference points to a file
    np.savetxt(reference_points_output_path, reference_points, fmt='%8.3f')
    print(f"Reference points saved to {reference_points_output_path}")

    # Create a grid within the defined box
    filtered_grid_points, unfiltered_grid_points = create_grid(min_coord, max_coord, step=1.0, exclusion_distance=0.8)
    print(f"Generated {len(filtered_grid_points)} filtered grid points.")
    print(f"Generated {len(unfiltered_grid_points)} unfiltered grid points.")

    # Load the trained model
    model = load_trained_model(model_path)
    print("Loaded trained model.")

    all_predictions = []    

    # Generate predictions for each grid point
    num_filtered_grid_points = len(filtered_grid_points)
    for i, grid_point in enumerate(filtered_grid_points):
        print(f"Processing filtered grid point {i+1}/{num_filtered_grid_points}...")
        closest_atoms = find_closest_atoms(grid_point, atoms, num_closest=10)
        features = prepare_features(closest_atoms)
        prediction = make_predictions(model, features)
        all_predictions.append((grid_point[0], grid_point[1], grid_point[2], prediction[0][0]))

    all_predictions = sorted(all_predictions, key=lambda x: -x[3])  # Sort by probability

    coords, probs = zip(*[(pred[:3], pred[3]) for pred in all_predictions])

    # Save all predictions to PDB file
    save_predictions_to_pdb(probs, coords, pdb_output_path_all)

    # Save confident predictions to PDB file
    save_predictions_to_pdb(probs, coords, pdb_output_path_confident, filter_threshold=0.3)

    # Cluster predictions and save to PDB file
    grid_size = 3.0  # Assuming a 3x3 mini grid size for clustering
    clustered_coords, clustered_probs = cluster_predictions(coords, probs, grid_size)
    save_predictions_to_pdb(clustered_probs, clustered_coords, pdb_output_path_clustered)

    print("Prediction process completed.")
