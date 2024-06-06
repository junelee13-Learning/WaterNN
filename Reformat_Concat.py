import sys

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
        'occupancy': line[54:60].strip(),
        'temperature_factor': line[60:66].strip(),
        'element_symbol': line[76:78].strip(),
        'charge': line[78:80].strip()
    }

def format_pdb_line(entry, new_atom_serial, new_residue_sequence_number):
    """
    Formats a dictionary of parsed PDB line entries back into a standard PDB line format with new serial numbers.
    
    Parameters:
    entry (dict): A dictionary containing the parsed entries.
    new_atom_serial (int): The new atom serial number.
    new_residue_sequence_number (int): The new residue sequence number.

    Returns:
    str: A formatted string in standard PDB line format.
    """
    return "{:<6}{:>5} {:<4}{:<1}{:<3} {:<1}{:>4}{:<1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6}{:>6}          {:>2}{:>2}".format(
        entry['record_name'],
        new_atom_serial,
        entry['atom_name'],
        entry['alternate_location_indicator'],
        entry['residue_name'],
        entry['chain_identifier'],
        new_residue_sequence_number,
        entry['code_for_insertion_of_residues'],
        entry['x_coordinate'],
        entry['y_coordinate'],
        entry['z_coordinate'],
        entry['occupancy'],
        entry['temperature_factor'],
        entry['element_symbol'],
        entry['charge']
    )

def reorder_pdb(pdb_lines):
    """
    Reorders the atoms and residues in a list of PDB lines.
    
    Parameters:
    pdb_lines (list): A list of PDB lines.

    Returns:
    list: A list of reordered PDB lines.
    """
    parsed_lines = [parse_pdb_line(line) for line in pdb_lines]
    reordered_lines = []
    
    current_residue_sequence_number = 1
    current_atom_serial = 1
    previous_residue = None
    
    for entry in parsed_lines:
        residue_key = (entry['residue_name'], entry['chain_identifier'], entry['residue_sequence_number'])
        
        if previous_residue is None or residue_key != previous_residue:
            previous_residue = residue_key
            current_residue_sequence_number += 1

        formatted_line = format_pdb_line(entry, current_atom_serial, current_residue_sequence_number)
        reordered_lines.append(formatted_line)
        current_atom_serial += 1

    return reordered_lines

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reorder_pdb.py <pdb_file_path>")
        sys.exit(1)
    pdb_file_path = sys.argv[1]

    with open(pdb_file_path, 'r') as pdb_file:
        pdb_lines = [line.strip() for line in pdb_file if line.startswith("ATOM") or line.startswith("HETATM")]

    reordered_lines = reorder_pdb(pdb_lines)

    output_file_path = pdb_file_path.replace(".pdb", "_reordered.pdb")
    with open(output_file_path, 'w') as output_file:
        for line in reordered_lines:
            output_file.write(line + "\n")

    print(f"Reordered PDB file written to {output_file_path}")
