import os
import json
from pymatgen.io.vasp.outputs import Vasprun

# Path to the folder containing all calculation directories
root_dir = "/home/zahed/Desktop/ML_interview/data/calcs/Si-O_vasprunxml"  # Change to your parent folder path

# List to store extracted data
data_list = []

# Loop through all calculation folders
for i in range(1, 401):  # Assuming folder names are CalcFold1, CalcFold2, ..., CalcFold400
    filename = f"{i}_vasprun.xml"
    vasprun_file = os.path.join(root_dir, filename)

    try:
        # Load VASP results from vasprun.xml
        vasprun = Vasprun(vasprun_file)

        # Extract relevant data
        data = {
            "folder": vasprun_file,
            "energy": vasprun.final_energy,  # Total energy
            "forces": vasprun.ionic_steps[-1]["forces"],  # Forces on each atom
            "stresses": vasprun.ionic_steps[-1]["stress"],  # Stress tensor
            "positions": vasprun.ionic_steps[-1]["structure"].frac_coords.tolist(),  # Atomic positions
            "lattice": vasprun.final_structure.lattice.matrix.tolist(),  # Lattice vectors
            # "atom_types": vasprun.final_structure.species_string,  # Element types as string
            "atom_types": vasprun.ionic_steps[-1]["structure"].composition.as_dict(),
            "num_atoms": vasprun.final_structure.num_sites,  # Number of atoms
            "numbers" : vasprun.final_structure.atomic_numbers,
            "spgnum": vasprun.final_structure.get_space_group_info()[1],
        }

        # Append data to the list
        data_list.append(data)

        print(f"Extracted data from {filename}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# Save data to a JSON file
output_file = "Si-O_vasprunxml.json"
with open(output_file, "w") as f:
    json.dump(data_list, f, indent=4)

print(f"Data saved to {output_file}")
