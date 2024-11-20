#!/usr/bin/env python

import os
import sys
import numpy as np
import ase.io
from ase.build import sort
from random import sample
from numba import jit
import libfp
from scipy.optimize import linear_sum_assignment
from functools import reduce
from ase.data import chemical_symbols

def read_types(atoms):
    """
    Reads atomic types from an ASE Atoms object and returns an array of types.
    """
    atom_symbols = atoms.get_chemical_symbols()
    unique_symbols, counts = np.unique(atom_symbols, return_counts=True)

    types = []
    for i in range(len(unique_symbols)):
        types.extend([i + 1] * counts[i])  # Map atom type to integers starting from 1

    return np.array(types, dtype=int)

@jit(nopython=True)
def compute_cost_matrix(fp1, fp2, types, itype):
    """
    Optimized computation of the cost matrix (MX) for a given atomic type.
    """
    nat = len(fp1)
    MX = np.zeros((nat, nat))
    for iat in range(nat):
        if types[iat] == itype:
            for jat in range(nat):
                if types[jat] == itype:
                    tfpd = fp1[iat] - fp2[jat]
                    MX[iat][jat] = np.sqrt(np.sum(tfpd * tfpd))
    return MX

def get_fp_dist(fp1, fp2, types):
    ntyp = len(set(types))
    nat = len(fp1)
    fpd = 0.0
    
    for ityp in range(ntyp):
        itype = ityp + 1
        MX = compute_cost_matrix(fp1, fp2, types, itype)
        row_ind, col_ind = linear_sum_assignment(MX)
        total = MX[row_ind, col_ind].sum()
        fpd += total

    fpd = fpd / nat
    return fpd

@jit('(float64)(float64[:,:], int32, int32[:])', nopython=True)
def get_fpe(fp, ntyp, types):
    nat = len(fp)
    e = 0.
    fp = np.ascontiguousarray(fp)
    for ityp in range(ntyp):
        itype = ityp + 1
        e0 = 0.
        for i in range(nat):
            for j in range(nat):
                if types[i] == itype and types[j] == itype:
                    vij = fp[i] - fp[j]
                    t = np.vdot(vij, vij)
                    e0 += t
            e0 += 1.0 / (np.linalg.norm(fp[i]) ** 2)
        e += e0
    return e

def get_fp_mat(atoms, cutoff=4.0, contract=False, lmax=0, nx=400):
    lat = atoms.cell[:]
    rxyz = atoms.get_positions()
    chem_nums = list(atoms.numbers)
    znucl_list = reduce(lambda re, x: re + [x] if x not in re else re, chem_nums, [])
    ntyp = len(znucl_list)
    znucl = np.array(znucl_list, np.int32)
    types = read_types(atoms)

    cell = (lat, rxyz, types, znucl)
    nx = np.int32(nx)
    lmax = np.int32(lmax)
    cutoff = np.float64(cutoff)

    if lmax == 0:
        lseg = 1
        orbital = 's'
    else:
        lseg = 4
        orbital = 'sp'

    if len(rxyz) == len(types) and len(set(types)) == len(znucl):
        if contract:
            fp = libfp.get_sfp(cell, cutoff=cutoff, natx=nx, log=False, orbital=orbital)
            tmp_fp = []
            for i in range(len(fp)):
                if len(fp[i]) < 20:
                    tmp_fp_at = fp[i].tolist() + [0.0] * (20 - len(fp[i]))
                    tmp_fp.append(tmp_fp_at)
            fp = np.array(tmp_fp, dtype=np.float64)
        else:
            fp = libfp.get_lfp(cell, cutoff=cutoff, natx=nx, log=False, orbital=orbital)

    return fp

def fermi_dirac(x, T=0.01):
    """
    Fermi-Dirac-like distribution function with temperature parameter.
    
    Parameters:
    - x: Energy difference
    - T: Temperature parameter (controls the sharpness of transition)
         Lower T = sharper transition = more selective
         Higher T = smoother transition = more accepting
    """
    return 1 / (1 + np.exp(x/T))

def metropolis_hastings(origin_fpe, old_fpe, new_fpe, amplification=1000.0):
    """
    Metropolis-Hastings acceptance criterion using original structure as reference.
    
    Parameters:
    - origin_fpe: fingerprint energy of original structure (reference)
    - old_fpe: fingerprint energy of current structure
    - new_fpe: fingerprint energy of proposed structure
    - amplification: factor to amplify small FPE differences
    """
    
    scaled_delta = (new_fpe - old_fpe)/origin_fpe * amplification
    acceptance_prob = fermi_dirac(scaled_delta, T=0.1)
    
    return np.random.rand() < acceptance_prob

def POSCAR_GEN_MCMC(atoms_origin, elem_from, elem_to, max_subs, max_structures, max_iter=5000):
    """
    Generate structures using the MCMC process with the Metropolis-Hastings criterion to penalize
    low-symmetry structures. The function generates random substitutions of atoms and accepts or
    rejects them based on the fingerprint energy.
    """
    natom = len(atoms_origin)
    from_indices = []

    # Collect indices of atoms to be substituted (element 'from')
    for i_at in range(natom):
        if atoms_origin[i_at].symbol == elem_from:
            from_indices.append(i_at)

    accepted_structures = []

    # Calculate the fingerprint energy for the original structure
    origin_fp = get_fp_mat(atoms=atoms_origin)
    origin_types = np.int32(read_types(atoms_origin))
    origin_fpe = get_fpe(origin_fp, len(set(origin_types)), origin_types)

    # Loop over the number of substitutions (1 to max_subs)
    for i_C in range(max_subs):
        icount = 1
        n_C = i_C + 1

        # Restart with the original atoms for each n_C
        old_atoms = atoms_origin.copy()
        old_fp = get_fp_mat(atoms=old_atoms)
        old_types = np.int32(read_types(old_atoms))
        old_fpe = get_fpe(old_fp, len(set(old_types)), old_types)

        for i in range(max_iter):
            new_atoms = atoms_origin.copy()

            # Randomly select atoms to substitute
            subs_indices = sample(from_indices, n_C)
            for idx in subs_indices:
                new_atoms[idx].symbol = elem_to

            # Sort and get the new fingerprint matrix
            sorted_atoms = sort(new_atoms)
            new_fp = get_fp_mat(atoms=sorted_atoms)
            new_types = np.int32(read_types(sorted_atoms))
            new_fpe = get_fpe(new_fp, len(set(new_types)), new_types)

            if icount == 1:
                # First substitution step: always accept
                accepted_structures.append(sorted_atoms.copy())
                poscar_name = f'POSCAR_{n_C}_{icount}'
                ase.io.write(poscar_name, sorted_atoms, 'vasp', direct=True, long_format=True, vasp5=True)
                print(f"First substitution step: Structure accepted: {poscar_name}")

                old_atoms = sorted_atoms.copy()
                old_fp = new_fp.copy()
                old_fpe = new_fpe
                icount += 1
            else:
                # Apply Metropolis-Hastings using origin_fpe as reference
                if metropolis_hastings(origin_fpe, old_fpe, new_fpe):
                    accepted_structures.append(sorted_atoms.copy())
                    poscar_name = f'POSCAR_{n_C}_{icount}'
                    ase.io.write(poscar_name, sorted_atoms, 'vasp', direct=True, long_format=True, vasp5=True)
                    print(f"Structure accepted: {poscar_name}")

                    old_atoms = sorted_atoms.copy()
                    old_fp = new_fp.copy()
                    old_fpe = new_fpe
                    icount += 1

            if icount >= max_structures:
                break

    print(f"POSCAR_GEN_MCMC: Finished generating {len(accepted_structures)} structures.")

def POST_PROC(caldir):
    """
    Post-processing function that runs once all structures have been generated.
    """
    try:
        os.system('bash reduce_sim_struct.sh')
    except:
        os.system("rm -rf test_*")
        os.system("rm -rf POSCAR_*")
        raise Exception("There is something wrong with the first try. Double-check your aflow_test directory.")

if __name__ == '__main__':
    caldir = './'

    # Load POSCAR and extract element symbols present in the structure
    atoms_origin = ase.io.read(caldir + 'POSCAR')
    poscar_elements = set(atoms_origin.get_chemical_symbols())

    # Get user input
    elem_from = input("Enter the element to be substituted: ").capitalize()
    elem_to = input("Enter the new element: ").capitalize()

    # Check if the element to be substituted is present in the POSCAR file
    if elem_from not in poscar_elements:
        raise ValueError(f"The element '{elem_from}' is not present in the POSCAR file.")

    # Check if the new element is valid according to the ase.data.chemical_symbols module
    if elem_to not in chemical_symbols:
        raise ValueError("Invalid element! Please enter a valid element symbol for the new element.")

    # Get additional inputs
    max_subs = int(input("Enter the maximum number of atoms to substitute: "))
    max_structures = int(input("Enter the maximum number of structures per substitution: "))

    # Generate structures using POSCAR_GEN_MCMC
    POSCAR_GEN_MCMC(atoms_origin, elem_from, elem_to, max_subs, max_structures)

    # Ensure that POST_PROC waits for structure generation to finish
    print("All substitutions complete. Starting post-processing...")

    # Call post-processing function
    POST_PROC(caldir)