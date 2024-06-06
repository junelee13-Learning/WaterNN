import sys
import os

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
        'x_coordinate': line[30:38].strip(),
        'y_coordinate': line[38:46].strip(),
        'z_coordinate': line[46:54].strip(),
        'occupancy': line[54:60].strip(),
        'temperature_factor': line[60:66].strip(),
        'element_symbol': line[76:78].strip(),
        'charge': line[78:80].strip()
    }

def format_pdb_line(parsed_line):
    """
    Formats a dictionary of parsed PDB line entries back into a standard PDB line format.
    
    Parameters:
    parsed_line (dict): A dictionary containing the parsed entries.

    Returns:
    str: A formatted string in standard PDB line format.
    """
    return "{:<6}{:>5} {:<4}{:<1}{:<3} {:<1}{:>4}{:<1}   {:>8}{:>8}{:>8}{:>6}{:>6}          {:>2}{:>2}".format(
        parsed_line['record_name'],
        parsed_line['atom_serial_number'],
        parsed_line['atom_name'],
        parsed_line['alternate_location_indicator'],
        parsed_line['residue_name'],
        parsed_line['chain_identifier'],
        parsed_line['residue_sequence_number'],
        parsed_line['code_for_insertion_of_residues'],
        parsed_line['x_coordinate'],
        parsed_line['y_coordinate'],
        parsed_line['z_coordinate'],
        parsed_line['occupancy'],
        parsed_line['temperature_factor'],
        parsed_line['element_symbol'],
        parsed_line['charge']
    )

def parse_pdb_file(file_path):
    """
    Parses a PDB file and breaks down each line into individual entries.
    
    Parameters:
    file_path (str): The path to the PDB file.

    Returns:
    list: A list of dictionaries containing the parsed entries for each water molecule.
    """
    parsed_lines = []

    with open(file_path, 'r') as pdb_file:
        for line in pdb_file:
            record_type = line[0:6].strip()
            residue_name = line[17:20].strip()
            if record_type == "HETATM" and residue_name == "HOH":
                parsed_line = parse_pdb_line(line)
                parsed_lines.append(parsed_line)

    return parsed_lines

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py input_pdb_file")
        sys.exit(1)
    pdb_file_path = sys.argv[1]
    parsed_pdb = parse_pdb_file(pdb_file_path)
    base_name, extension = os.path.splitext(os.path.basename(pdb_file_path))
    output_folder_path = "Parsed_PDB_Files"
    output_subfolder_path = os.path.join(output_folder_path, base_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path, exist_ok=True)
    output_file_path = os.path.join(output_subfolder_path, f"{base_name}_parsed_waters.pdb")
    with open(output_file_path, 'w') as output_file:
        for entry in parsed_pdb:
            formatted_line = format_pdb_line(entry)
            output_file.write(formatted_line + "\n")
    print(f"Parsed water molecules written to: {output_file_path}")