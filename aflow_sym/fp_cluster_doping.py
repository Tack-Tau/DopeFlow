#!/usr/bin/env python

import os
import sys
import numpy as np
import ase.io
from ase.build import sort
from random import sample, randint, random
from numba import jit
import libfp
from scipy.optimize import linear_sum_assignment
from functools import reduce
from ase.data import chemical_symbols
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from math import exp
from reformpy.entropy import calculate_entropy_jit

def read_types(atoms, znucl_list=None):
    """
    Reads atomic types from an ASE Atoms object and returns an array of types.
    
    Parameters:
    - atoms: ASE Atoms object
    - znucl_list: Optional pre-computed list of unique atomic numbers in order of appearance
    """
    if znucl_list is None:
        chem_nums = list(atoms.numbers)
        znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])

    types = np.array([znucl_list.index(n) + 1 for n in atoms.numbers], int)

    return types

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
    types = read_types(atoms, znucl_list)

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

def entropy_guided_mcmc_sampling(atoms_origin, from_indices, elem_to, n_C,
                                max_structures=10, n_iterations=10000, 
                                temperature=1.0, thin=10, burnin=1000):
    """
    Generate diverse substituted structures using entropy-guided MCMC sampling.
    Uses ReformPy's FP entropy to maximize atomic environment diversity.
    
    Algorithm:
    1. Start with random substitution configuration
    2. Propose swaps between substituted and non-substituted sites
    3. Accept/reject based on entropy change (Metropolis-Hastings criterion)
    4. Collect structures from equilibrated Markov chain
    
    Parameters:
    - atoms_origin: Original ASE Atoms object
    - from_indices: List of indices that can be substituted
    - elem_to: New element symbol
    - n_C: Number of substitutions
    - max_structures: Maximum number of diverse structures to return
    - n_iterations: Total MCMC iterations
    - temperature: MCMC temperature (higher = more exploration)
    - thin: Keep every nth sample (reduces correlation)
    - burnin: Number of initial iterations to discard
    
    Returns:
    - selected_structures: List of diverse structures
    - entropies: Corresponding entropy values
    """
    print(f"Entropy-guided MCMC sampling for {n_C} substitutions...")
    print(f"  Parameters: {n_iterations} iterations, temp={temperature}, thin={thin}, burnin={burnin}")
    
    # Initialize with random configuration
    current_indices = sorted(sample(from_indices, n_C))
    current_atoms = atoms_origin.copy()
    for idx in current_indices:
        current_atoms[idx].symbol = elem_to
    
    # Calculate initial fingerprint and entropy
    sorted_atoms = sort(current_atoms)
    current_fp = get_fp_mat(atoms=sorted_atoms)
    current_entropy = calculate_entropy_jit(current_fp, min_threshold=1e-8)
    
    # Track samples
    structures = []
    entropies = []
    substitution_patterns = []
    
    accepted = 0
    for iteration in range(n_iterations):
        # Propose swap: randomly select one substituted and one non-substituted site
        substituted_idx = randint(0, n_C - 1)
        old_site = current_indices[substituted_idx]
        
        # Get non-substituted sites
        non_substituted = [idx for idx in from_indices if idx not in current_indices]
        new_site = non_substituted[randint(0, len(non_substituted) - 1)]
        
        # Create proposed configuration
        proposed_indices = current_indices.copy()
        proposed_indices[substituted_idx] = new_site
        proposed_indices = sorted(proposed_indices)
        
        # Create proposed structure
        proposed_atoms = atoms_origin.copy()
        for idx in proposed_indices:
            proposed_atoms[idx].symbol = elem_to
        
        # Calculate proposed fingerprint and entropy
        sorted_proposed = sort(proposed_atoms)
        proposed_fp = get_fp_mat(atoms=sorted_proposed)
        proposed_entropy = calculate_entropy_jit(proposed_fp, min_threshold=1e-8)
        
        # Metropolis-Hastings acceptance criterion
        # We want to MAXIMIZE entropy, so accept if entropy increases
        # or with probability exp(ΔS/T) if it decreases
        delta_entropy = proposed_entropy - current_entropy
        
        accept_prob = min(1.0, exp(delta_entropy / temperature))
        
        if random() < accept_prob:
            # Accept proposed move
            current_indices = proposed_indices
            current_atoms = proposed_atoms
            sorted_atoms = sorted_proposed
            current_fp = proposed_fp
            current_entropy = proposed_entropy
            accepted += 1
        
        # Collect samples after burnin, with thinning
        if iteration >= burnin and (iteration - burnin) % thin == 0:
            # Check if this pattern is new (avoid exact duplicates)
            pattern_tuple = tuple(current_indices)
            if pattern_tuple not in substitution_patterns:
                structures.append(sorted_atoms.copy())
                entropies.append(current_entropy)
                substitution_patterns.append(pattern_tuple)
                
                # Stop if we have enough unique structures
                if len(structures) >= max_structures * 3:  # Generate extra for diversity
                    break
    
    acceptance_rate = accepted / n_iterations
    print(f"  MCMC acceptance rate: {acceptance_rate:.2%}")
    print(f"  Collected {len(structures)} unique structures")
    
    # Select top max_structures by entropy (most diverse)
    if len(structures) > max_structures:
        sorted_indices = np.argsort(entropies)[::-1]  # Descending order
        selected_indices = sorted_indices[:max_structures]
        selected_structures = [structures[i] for i in selected_indices]
        selected_entropies = [entropies[i] for i in selected_indices]
    else:
        selected_structures = structures
        selected_entropies = entropies
    
    print(f"  Selected {len(selected_structures)} structures")
    print(f"  Entropy range: [{min(selected_entropies):.4f}, {max(selected_entropies):.4f}]")
    
    return selected_structures, selected_entropies

def filter_by_kim_energy(structures, model_name, percentile=80):
    """
    Filter structures by KIM empirical potential energy.
    Excludes structures above the specified energy percentile.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - model_name: KIM model name for energy calculation
    - percentile: Energy percentile threshold (default 80, excludes top 20%)
    
    Returns:
    - filtered_indices: Indices of structures below the energy threshold
    - energies: Array of computed energies for all structures
    """
    from ase.calculators.kim.kim import KIM
    
    n_structures = len(structures)
    energies = np.zeros(n_structures)
    
    print(f"Computing KIM energies using model: {model_name}")
    
    # Create KIM calculator
    calc = KIM(model_name)
    
    # Compute energies for all structures
    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            energy = atoms_copy.get_potential_energy()
            # Normalize by number of atoms
            energies[i] = energy / len(atoms_copy)
        except Exception as e:
            print(f"Warning: Failed to compute energy for structure {i}: {e}")
            energies[i] = np.inf
    
    # Calculate percentile threshold
    energy_threshold = np.percentile(energies[np.isfinite(energies)], percentile)
    
    # Filter structures
    filtered_indices = [i for i in range(n_structures) if energies[i] <= energy_threshold]
    
    print(f"Energy threshold at {percentile}th percentile: {energy_threshold:.4f} eV/atom")
    print(f"Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}] eV/atom")
    print(f"Kept {len(filtered_indices)}/{n_structures} structures below threshold")
    
    return filtered_indices, energies

def visualize_entropy_distribution(entropies, n_substitutions, output_file=None):
    """
    Visualize the entropy distribution of generated structures.
    
    Parameters:
    - entropies: List of entropy values
    - n_substitutions: Number of substitutions
    - output_file: Output filename (default: entropy_distribution_{n_substitutions}.png)
    """
    if output_file is None:
        output_file = f'entropy_distribution_{n_substitutions}_substitutions.png'
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Entropy histogram
    plt.subplot(1, 2, 1)
    plt.hist(entropies, bins=min(30, len(entropies)), edgecolor='black', alpha=0.7)
    plt.xlabel('Entropy (FP Diversity Measure)')
    plt.ylabel('Count')
    plt.title(f'{n_substitutions} Substitutions: Entropy Distribution')
    plt.axvline(np.mean(entropies), color='r', linestyle='--', label=f'Mean: {np.mean(entropies):.3f}')
    plt.axvline(np.median(entropies), color='g', linestyle='--', label=f'Median: {np.median(entropies):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Entropy vs structure index (sorted)
    plt.subplot(1, 2, 2)
    sorted_indices = np.argsort(entropies)[::-1]
    sorted_entropies = [entropies[i] for i in sorted_indices]
    plt.plot(range(len(sorted_entropies)), sorted_entropies, 'o-', markersize=4)
    plt.xlabel('Structure Rank (by entropy)')
    plt.ylabel('Entropy')
    plt.title(f'{n_substitutions} Substitutions: Ranked Diversity')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'N={len(entropies)}\n'
    stats_text += f'Range=[{min(entropies):.3f}, {max(entropies):.3f}]\n'
    stats_text += f'Std={np.std(entropies):.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved entropy distribution plot to: {output_file}")

def POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, max_iter=10000, 
                       visualize=False, mcmc_temperature=1.0, kim_model=None, energy_percentile=80):
    """
    Generate diverse structures using entropy-guided MCMC sampling.
    Uses ReformPy's fingerprint entropy to maximize atomic environment diversity.
    
    Parameters:
    - atoms_origin: Original ASE Atoms object
    - elem_from: Element to substitute
    - elem_to: New element
    - max_subs: Maximum number of substitutions
    - max_structures: Maximum number of structures per substitution level
    - max_iter: Maximum MCMC iterations per substitution level (default: 10000)
    - visualize: Whether to generate entropy distribution plots
    - mcmc_temperature: MCMC temperature for entropy sampling (default: 1.0)
    - kim_model: KIM model name for energy filtering (None to disable)
    - energy_percentile: Energy percentile threshold for filtering (default: 80)
    
    Returns:
    - List of selected diverse structures
    """
    natom = len(atoms_origin)
    from_indices = []

    # Collect indices of atoms to be substituted (element 'from')
    for i_at in range(natom):
        if atoms_origin[i_at].symbol == elem_from:
            from_indices.append(i_at)

    print(f"Found {len(from_indices)} {elem_from} atoms that can be substituted")
    
    if max_subs > len(from_indices):
        max_subs = len(from_indices)
        print(f"Limiting max substitutions to {max_subs} (total number of {elem_from} atoms)")

    final_structures = []

    # Loop over the number of substitutions (1 to max_subs)
    for i_C in range(max_subs):
        n_C = i_C + 1
        print(f"\n{'='*70}")
        print(f"Substitution level: {n_C} atom(s)")
        print(f"{'='*70}")
        
        # Use entropy-guided MCMC sampling to generate diverse structures
        structures_n_C, entropies_n_C = entropy_guided_mcmc_sampling(
            atoms_origin=atoms_origin,
            from_indices=from_indices,
            elem_to=elem_to,
            n_C=n_C,
            max_structures=max_structures,
            n_iterations=max_iter,
            temperature=mcmc_temperature,
            thin=10,
            burnin=max(1000, max_iter // 10)
        )
        
        if len(structures_n_C) == 0:
            print(f"Warning: No structures generated for {n_C} substitution(s)")
            continue
        
        # Apply KIM energy filtering if requested
        if kim_model is not None and len(structures_n_C) > 1:
            print(f"Filtering structures by KIM energy...")
            filtered_indices, energies = filter_by_kim_energy(
                structures_n_C,
                kim_model, 
                energy_percentile
            )
            structures_n_C = [structures_n_C[i] for i in filtered_indices]
            entropies_n_C = [entropies_n_C[i] for i in filtered_indices]
            print(f"After energy filtering: {len(structures_n_C)} structures remain")
        
        # Visualize entropy distribution if requested
        if visualize and len(structures_n_C) > 0:
            visualize_entropy_distribution(entropies_n_C, n_C)
        
        # Write selected structures to POSCAR files
        for i, struct in enumerate(structures_n_C):
            poscar_name = f'POSCAR_{n_C}_{i+1}'
            ase.io.write(poscar_name, struct, format='vasp', direct=True, vasp5=True)
            print(f"Wrote structure to: {poscar_name} (entropy={entropies_n_C[i]:.4f})")
            final_structures.append(struct)
                
    print(f"POSCAR_GEN_CLUSTER: Finished generating {len(final_structures)} diverse structures.")
    return final_structures

def POST_PROC(caldir):
    """
    Post-processing function - with entropy-guided MCMC approach, filtering is done
    during structure generation. No additional processing needed.
    """
    print("Post-processing complete. Diverse structures have been generated using entropy-guided MCMC.")
    # Structures are already diverse (high entropy) and optionally energy-filtered

if __name__ == '__main__':
    caldir = './'

    # Load POSCAR and extract element symbols present in the structure
    atoms_origin = ase.io.read(caldir + 'POSCAR')
    poscar_elements = set(atoms_origin.get_chemical_symbols())

    # Get user input
    print("Enter the element to be substituted: ", end="", flush=True)
    elem_from = input().capitalize()
    print("Enter the new element: ", end="", flush=True)
    elem_to = input().capitalize()

    # Check if the element to be substituted is present in the POSCAR file
    if elem_from not in poscar_elements:
        raise ValueError(f"The element '{elem_from}' is not present in the POSCAR file.")

    # Check if the new element is valid according to the ase.data.chemical_symbols module
    if elem_to not in chemical_symbols:
        raise ValueError("Invalid element! Please enter a valid element symbol for the new element.")

    # Get additional inputs
    print("Enter the maximum number of atoms to substitute: ", end="", flush=True)
    max_subs = int(input())
    print("Enter the maximum number of structures per substitution: ", end="", flush=True)
    max_structures = int(input())
    print("Enter MCMC temperature (default 1.0, higher=more exploration): ", end="", flush=True)
    temp_input = input().strip()
    mcmc_temperature = float(temp_input) if temp_input else 1.0
    print("Enter MCMC iterations per level (default 10000): ", end="", flush=True)
    iter_input = input().strip()
    max_iter = int(iter_input) if iter_input else 10000
    print("Use KIM energy filtering? (y/n): ", end="", flush=True)
    use_kim = input().lower() in ['y', 'yes']
    kim_model = "Tersoff_LAMMPS_Tersoff_1989_SiGe__MO_350526375143_004" if use_kim else None
    print("Generate entropy distribution plots? (y/n): ", end="", flush=True)
    vis_input = input().lower()
    visualize = vis_input == 'y' or vis_input == 'yes'
    
    if visualize:
        print("Entropy distribution plots will be generated")
    
    if kim_model:
        print(f"KIM energy filtering enabled using model: {kim_model}")
    
    print(f"MCMC parameters: temperature={mcmc_temperature}, iterations={max_iter}")
    
    # Generate structures using entropy-guided MCMC approach
    POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, 
                       max_iter=max_iter, visualize=visualize, 
                       mcmc_temperature=mcmc_temperature, kim_model=kim_model)

    # Call post-processing function
    print("All substitutions complete. Starting post-processing...")
    POST_PROC(caldir)