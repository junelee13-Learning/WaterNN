import sys
import os
import math
import numpy as np
from scipy.spatial import distance
import shutil

# Classification of amino acids
charged_amino_acids = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
polar_amino_acids = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS', 'SEC'}  # Include selenocysteine
non_polar_amino_acids = {'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}

# Updated atomic radii for common elements (in Å)
atomic_radii = {
    'H': 0,           # Hydrogen (2.08 Å, but set to 0 due to molecular interactions)
    'C': 0.91,        # Carbon
    'N': 0.92,        # Nitrogen
    'O': 0.65,        # Oxygen
    'S': 1.27,        # Sulfur
    'Se': 1.4        # Selenium (same as Sulfur)
}

# Dictionary mapping amino acids to their assigned numbers based on their types
amino_acid_numbers = {}
for aa in charged_amino_acids:
    amino_acid_numbers[aa] = 200
for aa in polar_amino_acids:
    amino_acid_numbers[aa] = 100
for aa in non_polar_amino_acids:
    amino_acid_numbers[aa] = 10

# Any other amino acid type gets 0
all_amino_acids = charged_amino_acids | polar_amino_acids | non_polar_amino_acids
remaining_amino_acids = set(sorted(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC'])) - all_amino_acids
for aa in remaining_amino_acids:
    amino_acid_numbers[aa] = 0

def parse_pdb_line(line):
    """
    Parses a line from a PDB file into its individual entries.
    
    Parameters:
    line (str): A single line from a PDB file.

    Returns:
    dict: A dictionary containing the parsed entries.
    """
    return {
        'record_name': line[0:6].strip(),
        'atom_serial_number': line[6:11].strip(),
        'atom_name': line[12:16].strip(),
        'alternate_location_indicator': line[16:17].strip(),
        'residue_name': line[17:20].strip(),
        'chain_identifier': line[21:22].strip(),
        'residue_sequence_number': line[22:26].strip(),
        'code_for_insertion_of_residues': line[26:27].strip(),
        'x_coordinate': float(line[30:38].strip()),
        'y_coordinate': float(line[38:46].strip()),
        'z_coordinate': float(line[46:54].strip()),
        'occupancy': float(line[54:60].strip()) if line[54:60].strip() else 1.0,
        'temperature_factor': line[60:66].strip(),
        'element_symbol': line[76:78].strip(),
        'charge': line[78:80].strip(),
        'original_line': line.strip()  # Storing the original line
    }

def parse_pdb_file(file_path):
    """
    Parses a PDB file and breaks down each line into individual entries.
    
    Parameters:
    file_path (str): The path to the PDB file.

    Returns:
    list: A list of dictionaries containing the parsed entries.
    """
    parsed_lines = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parsed_line = parse_pdb_line(line)
                parsed_lines.append(parsed_line)

    return parsed_lines

def calculate_distance(atom1, atom2):
    """
    Calculates the distance between two atoms.
    
    Parameters:
    atom1 (dict): A dictionary containing the parsed entries for the first atom.
    atom2 (dict): A dictionary containing the parsed entries for the second atom.

    Returns:
    float: The distance between the two atoms.
    """
    return math.sqrt(
        (atom1['x_coordinate'] - atom2['x_coordinate']) ** 2 +
        (atom1['y_coordinate'] - atom2['y_coordinate']) ** 2 +
        (atom1['z_coordinate'] - atom2['z_coordinate']) ** 2
    )

def create_sphere(reference_point, radius, step=1.0):
    """
    Creates a sphere of given radius around the reference point, divided into smaller volumes.
    
    Parameters:
    reference_point (dict): A dictionary containing the coordinates of the reference point.
    radius (float): The radius of the sphere.
    step (float): The step size for dividing the sphere.

    Returns:
    list: A list of coordinates representing the centers of the mini spheres.
    """
    x0, y0, z0 = reference_point['x_coordinate'], reference_point['y_coordinate'], reference_point['z_coordinate']
    mini_spheres = []
    for x in np.arange(x0 - radius, x0 + radius, step):
        for y in np.arange(y0 - radius, y0 + radius, step):
            for z in np.arange(z0 - radius, z0 + radius, step):
                if math.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) <= radius:
                    mini_spheres.append((x, y, z))
    return mini_spheres

def calculate_intersection_volume(radius1, radius2, distance):
    """
    Calculate the volume of intersection between two spheres.
    
    Parameters:
    radius1 (float): Radius of the first sphere.
    radius2 (float): Radius of the second sphere.
    distance (float): Distance between the centers of the two spheres.
    
    Returns:
    float: Volume of intersection between the two spheres.
    """
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
    """
    Calculate the intrusion of residues within mini spheres.
    
    Parameters:
    mini_spheres (list): A list of coordinates representing the centers of the mini spheres.
    atoms (list): A list of dictionaries containing the parsed entries for the atoms.
    mini_sphere_radius (float): The radius of the mini spheres.
    
    Returns:
    dict: A dictionary where keys are mini sphere coordinates and values are the intrusion fractions of residues.
    """
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

def output_sphere_pdb(reference_point, mini_spheres, output_folder, file_name, reference_b_factor=3.5, mini_sphere_volume=1, intrusions=None):
    """
    Outputs the sphere and mini spheres in PDB format.
    
    Parameters:
    reference_point (dict): A dictionary containing the coordinates of the reference point.
    mini_spheres (list): A list of coordinates representing the centers of the mini spheres.
    output_folder (str): The path to the output folder.
    file_name (str): The name of the output PDB file.
    reference_b_factor (float): The B-factor for the reference sphere.
    mini_sphere_volume (float): The volume of each mini sphere.
    intrusions (dict): A dictionary containing intrusion values for mini spheres.
    """
    pdb_lines = []
    atom_serial_number = 1

    # Add the reference point as an ATOM record with the specified B-factor
    pdb_lines.append(
        f"ATOM  {atom_serial_number:5d}  O   HOH A{int(reference_point['residue_sequence_number']):4d}    "
        f"{reference_point['x_coordinate']:8.3f}{reference_point['y_coordinate']:8.3f}{reference_point['z_coordinate']:8.3f}"
        f"  {reference_b_factor:6.2f}  0.00           O"
    )
    atom_serial_number += 1

    # Calculate the radius that gives the specified mini sphere volume
    radius = (3 * mini_sphere_volume / (4 * math.pi)) ** (1/3)

    for i, (x, y, z) in enumerate(mini_spheres):
        occupancy = sum(intrusions[(x, y, z)].values()) if intrusions else radius
        pdb_lines.append(
            f"ATOM  {atom_serial_number:5d}  X   MS  B{str(i+1).rjust(4)}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:6.2f}  0.00           X"
        )
        atom_serial_number += 1

    # Add a remark for transparency, though not all PDB viewers will recognize this
    pdb_lines.append("REMARK 350   TRANSPARENCY 0.9")

    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w') as pdb_file:
        pdb_file.write("\n".join(pdb_lines) + "\n")

    print(f"PDB file for sphere and mini spheres written to {output_path}")

def output_intrusion_data(mini_spheres_folder, intrusions):
    """
    Outputs the intrusion data for each mini sphere into a separate file.
    
    Parameters:
    mini_spheres_folder (str): The path to the mini spheres folder.
    intrusions (dict): A dictionary containing intrusion values for mini spheres.
    """
    for i, (mini_sphere, residues) in enumerate(intrusions.items()):
        has_nonzero_intrusion = any(volume_fraction > 0 for elements in residues.values() for volume_fraction in elements.values())
        file_name = f'intrusion_{i+1}.txt'
        if has_nonzero_intrusion:
            file_name = f'intrusion_{i+1}_nonzero.txt'
        intrusion_file_path = os.path.join(mini_spheres_folder, file_name)
        with open(intrusion_file_path, 'w') as intrusion_file:
            for residue, elements in residues.items():
                residue_total_intrusion = sum(elements.values())
                intrusion_file.write(f"Residue: {residue}, Total Intrusion: {residue_total_intrusion:.4f}\n")
                for element, volume_fraction in elements.items():
                    intrusion_file.write(f"  Element: {element}, Volume Fraction: {volume_fraction:.4f}\n")

def output_intrusion_summary(mini_spheres_folder, intrusions, microsphere_number):
    """
    Outputs a summary file with the total intrusion per microsphere, including contributing amino acids.
    
    Parameters:
    mini_spheres_folder (str): The path to the mini spheres folder.
    intrusions (dict): A dictionary containing intrusion values for mini spheres.
    microsphere_number (int): The microsphere number.
    """
    summary_file_path = os.path.join(mini_spheres_folder, f'intrusion_summary_{microsphere_number}.txt')
    with open(summary_file_path, 'w') as summary_file:
        for i, (mini_sphere, residues) in enumerate(intrusions.items()):
            total_intrusion = sum(volume_fraction for elements in residues.values() for volume_fraction in elements.values())
            summary_file.write(f"Microsphere {i+1}: Total Intrusion: {total_intrusion:.4f}\n")
            for residue, elements in residues.items():
                residue_total_intrusion = sum(elements.values())
                summary_file.write(f"  Residue: {residue}, Total Intrusion: {residue_total_intrusion:.4f}\n")
                for element, volume_fraction in elements.items():
                    summary_file.write(f"    Element: {element}, Volume Fraction: {volume_fraction:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python residue_nearby.py <original_pdb_file_path> <water_pdb_file_path>")
        sys.exit(1)
    original_pdb_file_path = sys.argv[1]
    water_pdb_file_path = sys.argv[2]
    radius = 3.5  # angstroms

    # Create output directories
    water_file_base = os.path.basename(water_pdb_file_path).replace(".pdb", "")
    output_folder = water_file_base

    # Check if the output directory exists
    if os.path.exists(output_folder):
        user_input = input(f"The directory '{output_folder}' already exists. Do you want to delete it and proceed? (yes/no): ")
        if user_input.lower() == 'yes':
            shutil.rmtree(output_folder)
            print(f"Deleted directory '{output_folder}'.")
        else:
            print("Aborting operation.")
            sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    macrospheres_folder = os.path.join(output_folder, "macrospheres")
    microspheres_folder = os.path.join(output_folder, "microspheres")
    os.makedirs(macrospheres_folder, exist_ok=True)
    os.makedirs(microspheres_folder, exist_ok=True)

    # Create Data folder
    data_folder = os.path.join("Data", water_file_base, "intrusiondata")
    
    # Check if the Data/name directory exists
    if os.path.exists(data_folder):
        user_input = input(f"The directory '{data_folder}' already exists. Do you want to delete it and proceed? (yes/no): ")
        if user_input.lower() == 'yes':
            shutil.rmtree(data_folder)
            print(f"Deleted directory '{data_folder}'.")
        else:
            print("Aborting operation.")
            sys.exit(1)

    os.makedirs(data_folder, exist_ok=True)

    original_atoms = parse_pdb_file(original_pdb_file_path)
    water_atoms = parse_pdb_file(water_pdb_file_path)

    water_oxygen_atoms = [atom for atom in water_atoms if atom['atom_name'] == 'O']

    for water_oxygen_atom in water_oxygen_atoms:
        mini_spheres = create_sphere(water_oxygen_atom, radius, step=1.0)

        # Calculate intrusions
        mini_sphere_radius = (3 * 1 / (4 * math.pi)) ** (1/3)
        intrusions = calculate_intrusion(mini_spheres, original_atoms, mini_sphere_radius)

        # Debug output
        debug_output_file_path = os.path.join(output_folder, f'tagged_microspheres_debug_{water_oxygen_atom["residue_sequence_number"]}.txt')
        with open(debug_output_file_path, 'w') as debug_output_file:
            for i, (mini_sphere, residues) in enumerate(intrusions.items()):
                debug_output_file.write(f"Microsphere {i+1} {mini_sphere}: {residues}\n")

        print(f"Debug output written to {debug_output_file_path}")

        # Output the macrosphere in PDB format
        output_sphere_pdb(water_oxygen_atom, mini_spheres, macrospheres_folder, file_name=f'macrosphere_{water_oxygen_atom["residue_sequence_number"]}.pdb', reference_b_factor=3.5)

        # Output each microsphere in PDB format
        microsphere_folder = os.path.join(microspheres_folder, f"microspheres_{water_oxygen_atom['residue_sequence_number']}")
        os.makedirs(microsphere_folder, exist_ok=True)
        for i, mini_sphere in enumerate(mini_spheres):
            mini_sphere_atoms = {
                'x_coordinate': mini_sphere[0],
                'y_coordinate': mini_sphere[1],
                'z_coordinate': mini_sphere[2],
                'residue_sequence_number': i + 1
            }
            output_sphere_pdb(mini_sphere_atoms, [], microsphere_folder, file_name=f'microsphere_{i+1}.pdb', intrusions=intrusions)

        # Output intrusion data
        output_intrusion_data(microsphere_folder, intrusions)
        
        # Output intrusion summary
        output_intrusion_summary(data_folder, intrusions, water_oxygen_atom['residue_sequence_number'])
