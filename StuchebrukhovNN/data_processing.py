import os
import sys
import math
import csv

# Define the atom types for binary tagging
atom_types = {'H': 0, 'C': 1, 'O': 2, 'N': 3, 'S': 4, 'Others': 5}

# Define the amino acids and other residue types for binary tagging
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
other_residues = ['CU', 'MG', 'SE', 'HEA', 'TRP', 'TRD', 'DMU', 'Others']
residue_types = {aa: i for i, aa in enumerate(amino_acids)}
residue_types.update({res: len(amino_acids) + i for i, res in enumerate(other_residues)})

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

def find_closest_atoms(water_atoms, protein_atoms, num_closest=10):
    closest_atoms = {}
    for water_atom in water_atoms:
        distances = [
            (calculate_distance(water_atom, protein_atom), protein_atom) 
            for protein_atom in protein_atoms 
            if not (
                water_atom['atom_name'] == protein_atom['atom_name'] and 
                water_atom['residue_name'] == protein_atom['residue_name'] and 
                water_atom['residue_sequence_number'] == protein_atom['residue_sequence_number'] and 
                water_atom['chain_identifier'] == protein_atom['chain_identifier']
            )
        ]
        distances.sort(key=lambda x: x[0])
        closest_atoms[water_atom['atom_serial_number']] = distances[:num_closest]
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

def output_closest_atoms_data(output_folder, closest_atoms):
    for water_atom_serial, atoms in closest_atoms.items():
        water_folder = os.path.join(output_folder, str(water_atom_serial))
        os.makedirs(water_folder, exist_ok=True)
        
        for i, (_, atom) in enumerate(atoms):
            order = i + 1
            atom_folder_name = f"{order}_{atom['atom_name']}_{atom['residue_name']}{atom['residue_sequence_number']}{atom['chain_identifier']}"
            atom_folder = os.path.join(water_folder, atom_folder_name)
            os.makedirs(atom_folder, exist_ok=True)
            file_path = os.path.join(atom_folder, 'coordinates.csv')
            
            binary_tags = get_binary_tags(atom['element_symbol'])
            residue_tags = get_residue_tags(atom['residue_name'])
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Hydrogen', 'Carbon', 'Oxygen', 'Nitrogen', 'Sulfur', 'Others'] + amino_acids + other_residues + ['X', 'Y', 'Z'])
                writer.writerow(binary_tags + residue_tags + [atom['x_coordinate'], atom['y_coordinate'], atom['z_coordinate']])

def main(protein_pdb_path, water_pdb_path, output_folder):
    protein_atoms = parse_pdb_file(protein_pdb_path)
    water_atoms = parse_pdb_file(water_pdb_path)
    closest_atoms = find_closest_atoms(water_atoms, protein_atoms)
    output_closest_atoms_data(output_folder, closest_atoms)
    print(f"Output written to {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <protein_pdb_path> <water_pdb_path>")
        sys.exit(1)
    
    protein_pdb_path = sys.argv[1]
    water_pdb_path = sys.argv[2]
    output_folder = "data"

    if not os.path.exists(protein_pdb_path):
        print(f"Protein PDB file not found: {protein_pdb_path}")
        sys.exit(1)

    if not os.path.exists(water_pdb_path):
        print(f"Water PDB file not found: {water_pdb_path}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    main(protein_pdb_path, water_pdb_path, output_folder)
