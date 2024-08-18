import os
import sys
import math
import csv
import numpy as np

# Define the atom types for binary tagging
# Include specific atom names for detailed classification
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
backbone_residue_index = len(amino_acids) + len(other_residues)  # New residue index for backbone atoms

def parse_pdb_line(line):
    return {
        'record_name': line[0:6].strip(),
        'atom_serial_number': int(line[6:11].strip()),
        'atom_name': line[12:16].strip(),
        'alternate_location_indicator': line[16:17].strip(),
        'residue_name': line[17:20].strip(),
        'chain_identifier': line[21:22].strip(),
        'residue_sequence_number': int(line[22:26].strip()),
        'code_for_insertion_of_residues': line[26:27].strip(),
        'x_coordinate': float(line[30:38].strip()),
        'y_coordinate': float(line[38:46].strip()),
        'z_coordinate': float(line[46:54].strip()),
        'occupancy': float(line[54:60].strip()) if line[54:60].strip() else 1.0,
        'temperature_factor': line[60:66].strip(),
        'element_symbol': line[76:78].strip(),
        'charge': line[78:80].strip(),
        'original_line': line.strip()
    }

def parse_pdb_file(file_path):
    parsed_lines = []
    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parsed_lines.append(parse_pdb_line(line))
    return parsed_lines

def calculate_distance(atom1, atom2):
    return math.sqrt(
        (atom1['x_coordinate'] - atom2['x_coordinate']) ** 2 +
        (atom1['y_coordinate'] - atom2['y_coordinate']) ** 2 +
        (atom1['z_coordinate'] - atom2['z_coordinate']) ** 2
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

def find_closest_atoms(target_atoms, reference_atoms, num_closest=10):
    closest_atoms = {}
    for target_atom in target_atoms:
        distances = [
            (calculate_distance(target_atom, ref_atom), ref_atom) 
            for ref_atom in reference_atoms 
            if ref_atom['element_symbol'] != 'H' and not (
                target_atom['atom_name'] == ref_atom['atom_name'] and 
                target_atom['residue_name'] == ref_atom['residue_name'] and 
                target_atom['residue_sequence_number'] == ref_atom['residue_sequence_number'] and 
                target_atom['chain_identifier'] == ref_atom['chain_identifier']
            )
        ]
        distances.sort(key=lambda x: x[0])
        closest_atoms[target_atom['atom_serial_number']] = distances[:num_closest]
    return closest_atoms

def get_binary_tags(atom_name):
    tags = [0] * len(atom_types)  # Initialize a list with the correct number of zeros
    if atom_name in atom_types:
        tags[atom_types[atom_name]] = 1
    else:
        tags[atom_types['Others']] = 1
    return tags

def get_residue_tags(residue_name, atom_name):
    tags = [0] * (len(amino_acids) + len(other_residues) + 1)  # Initialize a list with the correct number of zeros
    if residue_name in residue_types:
        tags[residue_types[residue_name]] = 1
    if atom_name in backbone_atoms:
        tags[backbone_residue_index] = 1
    return tags

def output_closest_atoms_data(output_folder, closest_atoms, reference_atoms, label):
    for target_atom_serial, atoms in closest_atoms.items():
        target_atom = next(atom for atom in reference_atoms if atom['atom_serial_number'] == target_atom_serial)
        atom_folder_name = f"{target_atom['atom_name']}_{target_atom['residue_name']}_{target_atom['residue_sequence_number']}"
        atom_folder = os.path.join(output_folder, atom_folder_name)
        os.makedirs(atom_folder, exist_ok=True)
        file_name = 'coordinates_1.csv' if label == 1 else 'coordinates_0.csv'
        file_path = os.path.join(atom_folder, file_name)
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['N', 'CA', 'C', 'O', 'OXT', 'CB', 'CG', 'CD', 'CE', 'CZ', 
                             'OG', 'SG', 'Others'] + amino_acids + other_residues + ['Backbone'] + 
                            ['xr', 'yr', 'zr', 'r', 'cos_theta', 'cos_phi'])
            for i, (_, atom) in enumerate(atoms):
                binary_tags = get_binary_tags(atom['atom_name'])
                residue_tags = get_residue_tags(atom['residue_name'], atom['atom_name'])
                xr = atom['x_coordinate'] - target_atom['x_coordinate']
                yr = atom['y_coordinate'] - target_atom['y_coordinate']
                zr = atom['z_coordinate'] - target_atom['z_coordinate']
                r = calculate_distance(target_atom, atom)
                
                cos_theta = cos_phi = 0  # Default value when calculation cannot be performed
                if i > 0:
                    prev_atom = atoms[i-1][1]
                    vec1 = calculate_vector(target_atom, prev_atom)
                    vec2 = calculate_vector(target_atom, atom)
                    cos_theta = calculate_cos_theta(vec1, vec2)
                    
                    if i > 1:
                        prev_prev_atom = atoms[i-2][1]
                        vec3 = calculate_vector(target_atom, prev_prev_atom)
                        cos_phi = calculate_cos_phi(vec1, vec2, vec3)
                writer.writerow(binary_tags + residue_tags + [xr, yr, zr, r, cos_theta, cos_phi])

def main(pdb_file_path, output_folder):
    atoms = parse_pdb_file(pdb_file_path)
    # CD Restriction line
    atoms = [atom for atom in atoms if atom['chain_identifier'] in ['A', 'C']]
    non_hydrogen_atoms = [atom for atom in atoms if atom['element_symbol'] != 'H']
    water_atoms = [atom for atom in non_hydrogen_atoms if atom['residue_name'] == 'HOH']
    non_water_atoms = [atom for atom in non_hydrogen_atoms if atom['residue_name'] != 'HOH']
    
    closest_water_atoms = find_closest_atoms(water_atoms, non_hydrogen_atoms)
    closest_non_water_atoms = find_closest_atoms(non_water_atoms, non_hydrogen_atoms)
    
    output_closest_atoms_data(output_folder, closest_water_atoms, water_atoms, label=1)
    output_closest_atoms_data(output_folder, closest_non_water_atoms, non_water_atoms, label=0)
    
    print(f"Output written to {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pdb_file_path>")
        sys.exit(1)
    
    pdb_file_path = sys.argv[1]
    output_folder = "data_AC"

    if not os.path.exists(pdb_file_path):
        print(f"PDB file not found: {pdb_file_path}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    main(pdb_file_path, output_folder)
