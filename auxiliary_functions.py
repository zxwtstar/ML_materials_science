from __future__ import annotations
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Vasprun


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
def get_data(filepath, type="maml"):
    data = []
    if type == "maml":
        data_all = load_json(filepath)
        for d in data_all:
            # Extract relevant data
            entry = {
                "energy": d["outputs"]["energy"],
                "forces": d["outputs"]["forces"],
                "stress": d["outputs"]["stress"],
                "structure" : Structure.from_dict(d["structure"])
            }
            # Append data to the list
            data.append(entry)

    elif type == "daics":
        data_all = load_json(filepath)
        for d in data_all:
            # Extract relevant data
            lattice = d["lattice"]
            species = d["numbers"]
            frac_coords = d["frac_coordinates"]
            entry = {
                "energy": d["energy"],
                "f_energy": d["f_energy"],
                "structure" : Structure(lattice, species, frac_coords)
            }
            # Append data to the list
            data.append(entry)
            
    elif type == "mp":
        data_all = load_json(filepath)
        for d in data_all["data"]:
            structure = Structure.from_dict(d[0])
            energy = d[1]
            entry = {
                "energy": energy,
                "structure" : structure,
            }
            # Append data to the list
            data.append(entry)
            
    elif type == "calcs":
        # Path to the folder containing all calculation vasprun.xml
        root_dir = "/home/zahed/Desktop/ML_interview/data/calcs/Si-O_vasprunxml"

        data = []
        # List all files in the directory
        for i in range(len(os.listdir(root_dir))):
            filename = f"{i + 1}_vasprun.xml"
            vasprun_file = os.path.join(root_dir, filename)
            
            # Load VASP results from vasprun.xml
            vasprun = Vasprun(vasprun_file)

            # Number of atoms in the structure
            num_atoms = len(vasprun.final_structure)

            # Extract relevant data
            tmp = {
                "path": filename,
                "energy": vasprun.final_energy,  # Total energy
                # "energy_per_atom": vasprun.final_energy / num_atoms,  # Energy per atom
                "forces": vasprun.ionic_steps[-1]["forces"],
                "stress": vasprun.ionic_steps[-1]["stress"],
                "structure": vasprun.final_structure
            }

            # Append data to the list
            data.append(tmp)
            
    return data

def model_assessment(data):
    """
    Assess model performance using MAE, MAD, RMSE, and scaled error.

    Parameters:
    - data (array-like): A 2D array where the first column is actual values, 
    and the second column is predicted values.
    """
    # Convert input data to a numpy array
    data = np.array(data)
    
    # Separate actual and predicted values
    actual = data[:, 0]
    predicted = data[:, 1]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Calculate MAE and MAD
    errors = actual - predicted
    mae = np.mean(np.abs(errors))  # Mean Absolute Error
    mad = np.mean(np.abs(actual - np.mean(actual)))  # Mean Absolute Deviation

    # Calculate Scaled Error
    scaled_error = mae / mad if mad != 0 else np.inf  # Avoid division by zero

    return mae, mad, rmse, scaled_error

def plot_correlations(data, xlable, ylable, model, per_atom=False):
    data = np.array(data)
    x_arr = data[:, 0]
    y_arr = data[:, 1]
    
    mae, mad, rmse, scaled_error = model_assessment(data)
    # print(f"== MAE of {model} prediction is {mae * 1000} meV/atom")
    # print(f"== MAD of {model} prediction is {mad * 1000} meV/atom")
    # print(f"== RMSE of {model} prediction is {rmse * 1000} meV/atom")
    # print(f"== Scaled Error of {model} prediction is {scaled_error}")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x_arr, y_arr, alpha=0.5)

    # Calculate common limits for x and y
    combined_min = min(np.min(x_arr), np.min(y_arr))
    combined_max = max(np.max(x_arr), np.max(y_arr))

    # Set equal x and y limits
    plt.xlim(combined_min, combined_max)
    plt.ylim(combined_min, combined_max)

    # Add x=y line
    plt.plot([combined_min, combined_max], [combined_min, combined_max], color='red', linestyle='--', label='y = x')

    # Add labels and legend
    plt.title(model)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    # plt.legend()
    if per_atom:
        legend_text = (f"RMSE = {rmse * 1000:.3f} meV/atom\n"
                    f"MAE = {mae * 1000:.3f} meV/atom\n"
                    f"Scaled Error = {scaled_error:.3f}")
    else:
        legend_text = (f"RMSE = {rmse:.3f} eV\n"
                    f"MAE = {mae:.3f} eV\n"
                    f"Scaled Error = {scaled_error:.3f}")    
    plt.legend(title=legend_text)
    plt.show()
    
def calculate_formation_energy(total_energy, composition, element_energies):
    """
    Calculate the enthalpy of formation (eV/atom) of a compound.

    Parameters:
    -----------
    total_energy : float
        The total energy of the compound (eV).
    composition : dict
        Dictionary where keys are elements (str) and values are their counts in the compound.
        Example: {"Si": 1, "O": 2} for SiO2.
    element_energies : dict
        Dictionary where keys are elements (str) and values are their total energies per atom (eV/atom).
        Example: {"Si": -5.42, "O": -4.96}.

    Returns:
    --------
    float
        Enthalpy of formation in eV/atom.
    """
    # Calculate the total reference energy from the elements
    total_ref_energy = sum(element_energies[element] * count for element, count in composition.items())

    # Calculate the total number of atoms in the compound
    total_atoms = sum(composition.values())

    # Calculate the formation energy per atom
    formation_energy_per_atom = (total_energy - total_ref_energy) / total_atoms

    return formation_energy_per_atom

def get_other_energies(energies):
    data_en = []
    data_en_p_a = []
    data_form_en = []
    for i in range(len(energies)):
        n = energies[i]["num_sites"]
        en = energies[i]["energy"]
        pred_en = energies[i]["predicted_energy"]
        en_p_a = energies[i]["energy"]/n
        pred_en_p_a = energies[i]["predicted_energy"]/n
        form_en = energies[i]["formation_energy"]
        pred_form_en = energies[i]["formation_energy_predicted"]
        data_en.append([en, pred_en])
        data_en_p_a.append([en_p_a, pred_en_p_a])
        data_form_en.append([form_en, pred_form_en])
        
    return data_en, data_en_p_a, data_form_en