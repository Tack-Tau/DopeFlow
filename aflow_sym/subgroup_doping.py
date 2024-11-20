import os
import warnings
import ase.io
from ase.data import chemical_symbols
from pyxtal import pyxtal

warnings.filterwarnings("ignore", category=DeprecationWarning)

def perform_substitution(parent_atoms, sub_gen, group_type, max_cell, min_cell, N_max, element_sub, ratio_list):
    # Create pyxtal object from the parent atoms
    parent_xtal = pyxtal()
    parent_xtal.from_seed(parent_atoms)

    total_child_count = 0

    for ratio in ratio_list:
        print(f"Performing substitution with ratio: {ratio}")
        # Perform atomic substitution with the given ratio
        xtal_sub = parent_xtal.substitute_1_2(element_sub, ratio=ratio, group_type=group_type, max_cell=max_cell, min_cell=min_cell, N_max=N_max)

        # Save generated structures
        for count, sub in enumerate(xtal_sub, total_child_count + 1):
            child_atoms = sub.to_ase(resort=True, center_only=False)  # Dropped add_vacuum=False
            struct_name = child_atoms.get_chemical_formula(mode='hill', empirical=False)
            ase.io.write(f"{struct_name}_{sub_gen}_{count}.vasp", child_atoms, direct=True, vasp5=True)

        total_child_count += len(xtal_sub)

    return total_child_count

def get_child_name(sub_gen, child_count):
    # Find all files in the current directory that match the pattern "_sub_gen_child_count.vasp"
    pattern = f"_{sub_gen}_{child_count}.vasp"
    files = os.listdir()

    # Search for a file that ends with the required pattern
    for file_name in files:
        if file_name.endswith(pattern):
            print(file_name)  # Print the file name before returning it
            return file_name

    # If no file is found, raise an error
    raise FileNotFoundError(f"No file found with the pattern '*{pattern}'")

def run_substitution_loop(initial_structure_file, elem_from, elem_to, group_type='t+k', max_cell=4, min_cell=0, N_max=16, generations=3, max_ratio=10):
    parent_atoms = ase.io.read(initial_structure_file, format='vasp')

    # Check if elem_from is in the parent_atoms
    if elem_from not in parent_atoms.get_chemical_symbols():
        raise ValueError(f"The element '{elem_from}' is not present in the structure.")

    # Check if elem_to is a valid element according to ase.data.chemical_symbols
    if elem_to not in chemical_symbols:
        raise ValueError(f"The element '{elem_to}' is not a valid element according to ASE chemical symbols.")

    element_sub = {elem_from: [elem_from, elem_to]}
    sub_gen = 1

    # Generate the ratio list using the max_ratio parameter
    ratio_list = [[i, 1] for i in range(max_ratio, 0, -1)]  # Generates [[10, 1], [9, 1], ..., [1, 1]]

    while sub_gen <= generations:
        print("Generation:", sub_gen)  # Print the generation number
        # Perform substitution for current generation with different ratios
        child_count = perform_substitution(parent_atoms, sub_gen, group_type, max_cell, min_cell, N_max, element_sub, ratio_list)

        # Loop through all generated child structures for further substitutions
        for i in range(1, child_count + 1):
            try:
                child_struct_file = get_child_name(sub_gen, i)
                parent_atoms = ase.io.read(child_struct_file, format='vasp')
            except FileNotFoundError as e:
                print(e)
                continue

        sub_gen += 1

if __name__ == "__main__":
    # Example usage of the script
    initial_structure = "POSCAR"  # Input VASP POSCAR file
    elem_from = "Si"  # Element to substitute
    elem_to = "Ge"  # New element
    run_substitution_loop(initial_structure, elem_from, elem_to, generations=3, max_ratio=10)

