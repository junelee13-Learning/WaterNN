import os
import sys
import math
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN
import shutil

# Updated atomic radii for common elements (in Å)
atomic_radii = {
    'H': 0,           # Hydrogen (2.08 Å, but set to 0 due to molecular interactions)
    'C': 0.91,        # Carbon
    'N': 0.92,        # Nitrogen
    'O': 0.65,        # Oxygen
    'S': 1.27,        # Sulfur
    'Se': 1.4         # Selenium
}

# List of amino acids for the detailed contributions
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'OTHERS']

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

def create_grid(min_coord, max_coord, step=1.0):
    print("Creating grid points within the defined box...")
    x_range = np.arange(min_coord[0], max_coord[0] + step, step)
    y_range = np.arange(min_coord[1], max_coord[1] + step, step)
    z_range = np.arange(min_coord[2], max_coord[2] + step, step)
    grid_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    return grid_points

def create_sphere(x0, y0, z0, radius, step=1.0):
    mini_spheres = []
    for x in np.arange(x0 - radius, x0 + radius, step):
        for y in np.arange(y0 - radius, y0 + radius, step):
            for z in np.arange(z0 - radius, z0 + radius, step):
                if math.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) <= radius:
                    mini_spheres.append((x, y, z))
    return mini_spheres

def calculate_distance(atom1, atom2):
    return math.sqrt(
        (atom1['x_coordinate'] - atom2['x_coordinate']) ** 2 +
        (atom1['y_coordinate'] - atom2['y_coordinate']) ** 2 +
        (atom2['z_coordinate'] - atom1['z_coordinate']) ** 2
    )

def calculate_intersection_volume(radius1, radius2, distance):
    if distance >= (radius1 + radius2):
        return 0
    elif distance <= abs(radius1 - radius2):
        return (4/3) * math.pi * min(radius1, radius2)**3
    else:
        part1 = (radius1**2) * math.acos((distance**2 + radius1**2 - radius2**2) / (2 * distance * radius1))
        part2 = (radius2**2) * math.acos((distance**2 + radius2**2 - radius1**2) / (2 * distance * radius2))
        part3 = 0.5 * math.sqrt((-distance + radius1 + radius2) * (distance + radius1 - radius2) * (distance - radius1 + radius2) * (distance + radius1 + radius2))
        return part1 + part2 - part3

def calculate_intrusion(mini_spheres, atoms, mini_sphere_radius):
    print("Calculating intrusions into mini spheres...")
    intrusions = {mini_sphere: {} for mini_sphere in mini_spheres}
    
    for mini_sphere in mini_spheres:
        for atom in atoms:
            element = atom['element_symbol']
            if element in atomic_radii:
                atom_radius = atomic_radii[element]
                distance_to_center = calculate_distance({
                    'x_coordinate': mini_sphere[0],
                    'y_coordinate': mini_sphere[1],
                    'z_coordinate': mini_sphere[2]
                }, atom)
                intersection_volume = calculate_intersection_volume(mini_sphere_radius, atom_radius, distance_to_center)
                residue = atom['residue_name']
                if residue not in intrusions[mini_sphere]:
                    intrusions[mini_sphere][residue] = {}
                if element not in intrusions[mini_sphere][residue]:
                    intrusions[mini_sphere][residue][element] = intersection_volume
                else:
                    intrusions[mini_sphere][residue][element] += intersection_volume

    # Normalize the intrusions by the volume of the mini sphere
    mini_sphere_volume = (4/3) * math.pi * (mini_sphere_radius ** 3)
    for mini_sphere in intrusions:
        for residue in intrusions[mini_sphere]:
            for element in intrusions[mini_sphere][residue]:
                intrusions[mini_sphere][residue][element] /= mini_sphere_volume

    return intrusions

def prepare_features(intrusions):
    features = []
    for mini_sphere in intrusions:
        total_intrusion = sum(volume_fraction for elements in intrusions[mini_sphere].values() for volume_fraction in elements.values())
        residue_contributions = [0.0] * len(amino_acids)
        for residue in intrusions[mini_sphere]:
            residue_total_intrusion = sum(intrusions[mini_sphere][residue].values())
            if residue in amino_acids[:-1]:
                idx = amino_acids.index(residue)
            else:
                idx = amino_acids.index('OTHERS')
            residue_contributions[idx] += residue_total_intrusion
        features.append([total_intrusion] + residue_contributions)
    return np.array(features)

def make_predictions(model, features):
    predictions = model.predict(features)
    return predictions

def save_predictions(predictions, coords, output_path):
    print("Saving pre-clustered predictions...")
    with open(output_path, 'w') as f:
        for i, (coord, prob) in enumerate(zip(coords, predictions)):
            x, y, z = coord
            x, y, z, prob = x.item(), y.item(), z.item(), prob.item()  # Convert arrays to scalars
            f.write(f"ATOM  {i+1:5d}  O   HOH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  {prob:6.4f}  0.00           O\n")
    print(f"Pre-clustered predictions saved to {output_path}")

def save_clustered_predictions(clusters, highest_probs, output_path):
    print("Saving clustered predictions...")
    with open(output_path, 'w') as f:
        for i, (coord, prob) in enumerate(zip(clusters, highest_probs)):
            x, y, z = coord
            x, y, z, prob = x.item(), y.item(), z.item(), prob.item()  # Convert arrays to scalars
            f.write(f"ATOM  {i+1:5d}  O   HOH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  {prob:6.4f}  0.00           O\n")
    print(f"Clustered predictions saved to {output_path}")

def cluster_predictions(coordinates, predictions, eps=1.0, min_samples=1):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coordinates, sample_weight=predictions.flatten())
    labels = db.labels_
    unique_labels = set(labels)
    clusters = []
    highest_probs = []
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        cluster_coords = coordinates[class_member_mask]
        cluster_preds = predictions[class_member_mask]
        highest_prob_index = np.argmax(cluster_preds)
        highest_prob_coord = cluster_coords[highest_prob_index]
        highest_prob = cluster_preds[highest_prob_index]
        clusters.append(highest_prob_coord)
        highest_probs.append(highest_prob)
    return np.array(clusters), np.array(highest_probs)

def load_trained_model(model_path):
    return load_model(model_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_prediction.py <pdb_file_path>")
        sys.exit(1)

    pdb_file_path = sys.argv[1]  # Path to the PDB file
    model_path = os.path.join('trained_modules', 'intrusion_model.keras')  # Path to the trained model
    output_folder = 'predictions'
    pre_clustered_output_path = os.path.join(output_folder, 'pre_clustered_predictions.pdb')
    clustered_output_path = os.path.join(output_folder, 'clustered_predictions.pdb')
    reference_points_output_path = os.path.join(output_folder, 'reference_points.txt')

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
    grid_points = create_grid(min_coord, max_coord, step=1.0)
    print(f"Generated {len(grid_points)} grid points.")

    # Load the trained model
    model = load_trained_model(model_path)
    print("Loaded trained model.")

    all_predictions = []    

    # Generate spheres and make predictions
    num_grid_points = len(grid_points)
    for i, grid_point in enumerate(grid_points):
        print(f"Processing grid point {i+1}/{num_grid_points}...")
        x0, y0, z0 = grid_point
        mini_spheres = create_sphere(x0, y0, z0, radius=3.5, step=1.0)

        mini_sphere_radius = (3 * 1 / (4 * math.pi)) ** (1/3)
        intrusions = calculate_intrusion(mini_spheres, atoms, mini_sphere_radius)
        features = prepare_features(intrusions)
        predictions = make_predictions(model, features)
        all_predictions.extend([(x, y, z, prob) for (x, y, z), prob in zip(mini_spheres, predictions)])

    all_predictions = sorted(all_predictions, key=lambda x: -x[3])  # Sort by probability

    coords, probs = zip(*[(pred[:3], pred[3]) for pred in all_predictions])

    # Save pre-clustered predictions
    save_predictions(probs, coords, pre_clustered_output_path)

    # Cluster predictions
    clusters, highest_probs = cluster_predictions(np.array(coords), np.array(probs), eps=1.0, min_samples=1)

    # Save clustered predictions
    save_clustered_predictions(clusters, highest_probs, clustered_output_path)

    print("Prediction process completed.")
