#!/usr/bin/env python

import os
import sys
import numpy as np
import ase.io
from ase.build import sort
from itertools import combinations
from random import sample
from math import comb
from ase.data import chemical_symbols


def POSCAR_GEN(caldir, elem_from, elem_to, max_subs, max_structures, excluded_elements):
    atoms_origin = ase.io.read(caldir + 'POSCAR')
    atoms = atoms_origin.copy()

    natom = len(atoms)
    n_elem_from = 0
    from_indices = []

    # Count atoms of the element to be substituted and store their indices
    for i_at in range(natom):
        if atoms[i_at].symbol == elem_from and atoms[i_at].symbol not in excluded_elements:
            n_elem_from += 1
            from_indices.append(i_at)

    # Check if the element to be substituted exists in the POSCAR file
    if n_elem_from == 0:
        raise ValueError(f"The element '{elem_from}' is not present in the POSCAR file, or all instances of it are excluded from substitution.")

    # Check if the maximum number of substitutions is valid
    if max_subs > n_elem_from:
        raise ValueError(f"The maximum number of substitutions ({max_subs}) cannot be larger than the number of '{elem_from}' atoms ({n_elem_from}) in the POSCAR file.")

    # Generate POSCAR files with substitutions
    for i_C in range(max_subs):
        icount = 0
        n_C = i_C + 1
        total_combinations = comb(len(from_indices), n_C)

        # Limit the number of structures per substitution
        count_max = min(total_combinations, max_structures)

        # Randomly sample combination indices
        selected_combinations = sample(list(combinations(from_indices, n_C)), count_max)

        for comb_set in selected_combinations:
            atoms = atoms_origin.copy()
            icount += 1
            poscar_name = f'POSCAR_{n_C}_{icount}'

            for index in comb_set:
                atoms.symbols[index] = elem_to
            sorted_atoms = sort(atoms)
            ase.io.write(poscar_name, sorted_atoms, 'vasp', direct=True, long_format=True, vasp5=True)


def POST_PROC(caldir):
    try:
        os.system('bash reduce_sim_struct.sh')
    except:
        os.system("rm -rf test_*")
        os.system("rm -rf POSCAR_*")
        POSCAR_GEN(caldir)
        os.system('bash reduce_sim_struct.sh')
        raise Exception("There is something wrong with the first try. Double-check your aflow_test directory.")


if __name__ == '__main__':
    caldir = './'

    # Prompt user for element to substitute, new element, max substitutions, and max structures
    elem_from = input("Enter the element to be substituted: ").capitalize()
    elem_to = input("Enter the new element: ").capitalize()

    # Check if the new element is valid according to the ase.data.chemical_symbols module
    if elem_to not in chemical_symbols:
        raise ValueError("Invalid element! Please enter a valid element symbol for the new element.")

    # Load POSCAR and extract element symbols present in the structure
    atoms_origin = ase.io.read(caldir + 'POSCAR')
    poscar_elements = set(atoms_origin.get_chemical_symbols())

    # Check if the element to be substituted is present in the POSCAR file
    if elem_from not in poscar_elements:
        raise ValueError(f"The element '{elem_from}' is not present in the POSCAR file.")

    # Prompt for excluded elements
    excluded_elements_input = input("Enter elements to be excluded from substitution (comma-separated, e.g., Na,Cl): ")
    excluded_elements = {elem.strip().capitalize() for elem in excluded_elements_input.split(',')}

    # Ensure the elements to be excluded are valid
    for elem in excluded_elements:
        if elem not in chemical_symbols:
            raise ValueError(f"Invalid excluded element: {elem}. Please enter valid element symbols.")

    max_subs = int(input("Enter the maximum number of atoms to substitute: "))
    max_structures = int(input("Enter the maximum number of structures per substitution: "))

    POSCAR_GEN(caldir, elem_from, elem_to, max_subs, max_structures, excluded_elements)
    POST_PROC(caldir)

