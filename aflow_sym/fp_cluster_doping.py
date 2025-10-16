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
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

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

def compute_fp_diff_matrices(structures, fps, types_list):
    """
    Compute fingerprint difference matrices for all pairs of structures.
    Uses Hungarian algorithm to align atoms before computing differences.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - fps: List of fingerprint matrices
    - types_list: List of type arrays
    
    Returns:
    - fp_diff_vectors: Array of flattened FP difference vectors [n_structures, nat * fp_len]
    """
    n_structures = len(structures)
    ntyp = len(set(types_list[0]))
    nat = len(fps[0])
    fp_len = fps[0].shape[1]
    
    # Store all pairwise difference vectors
    # We'll build a matrix where each row is a flattened difference from structure i to all others
    fp_diff_matrix = np.zeros((n_structures, n_structures * nat * fp_len))
    
    for i in range(n_structures):
        fp1 = fps[i]
        types = types_list[i]
        
        for j in range(n_structures):
            if i == j:
                # Self-difference is zero
                continue
                
            fp2 = fps[j]
            
            # Compute aligned difference using Hungarian algorithm
            aligned_diff = np.zeros_like(fp1)
            
            for ityp in range(ntyp):
                itype = ityp + 1
                MX = compute_cost_matrix(fp1, fp2, types, itype)
                row_ind, col_ind = linear_sum_assignment(MX)
                
                # Extract aligned differences for this type
                for row, col in zip(row_ind, col_ind):
                    if types[row] == itype:
                        aligned_diff[row] = fp1[row] - fp2[col]
            
            # Flatten and store
            start_idx = j * nat * fp_len
            end_idx = (j + 1) * nat * fp_len
            fp_diff_matrix[i, start_idx:end_idx] = aligned_diff.flatten()
    
    return fp_diff_matrix

def compute_pca_coordinates(fp_diff_vectors, n_components):
    """
    Compute PCA coordinates from fingerprint difference vectors.
    Uses Gramian matrix eigendecomposition.
    
    Parameters:
    - fp_diff_vectors: Array of flattened FP difference vectors [n_structures, feature_dim]
    - n_components: Number of principal components to keep
    
    Returns:
    - pc_coordinates: Principal component coordinates [n_structures, n_components]
    - explained_variance: Explained variance ratio for each component
    """
    n_structures = fp_diff_vectors.shape[0]
    
    # Create Gramian matrix: G = X @ X.T
    gramian = fp_diff_vectors @ fp_diff_vectors.T
    
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gramian)
    
    # Sort by descending eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top n_components
    n_components = min(n_components, n_structures)
    pc_coordinates = eigenvectors[:, :n_components]
    
    # Scale by square root of eigenvalues for proper PCA coordinates
    pc_coordinates = pc_coordinates * np.sqrt(np.maximum(eigenvalues[:n_components], 0))
    
    # Compute explained variance
    total_variance = np.sum(np.maximum(eigenvalues, 0))
    if total_variance > 0:
        explained_variance = np.maximum(eigenvalues[:n_components], 0) / total_variance
    else:
        explained_variance = np.zeros(n_components)
    
    return pc_coordinates, explained_variance

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

def select_diverse_structures(structures, fps, types_list, max_structures=10, min_cluster_size=2, 
                               n_pca_components=10, visualize=False, n_substitutions=None):
    """
    Select diverse structures using PCA on fingerprint differences and HDBSCAN clustering.
    Selects structures at centroids and half-radius points in PCA space.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - fps: List of fingerprint matrices
    - types_list: List of type arrays
    - max_structures: Maximum number of structures to select
    - min_cluster_size: Minimum size of a cluster in HDBSCAN
    - n_pca_components: Number of PCA components to use for clustering
    - visualize: Whether to generate visualization plots
    - n_substitutions: Number of substitutions for labeling plots
    
    Returns:
    - List of indices of selected structures
    """
    n_structures = len(structures)
    
    if n_structures <= max_structures:
        # If we have fewer structures than requested, return all
        return list(range(n_structures))
    
    # Compute fingerprint difference matrices
    print(f"Computing FP difference matrices for {n_structures} structures...")
    fp_diff_vectors = compute_fp_diff_matrices(structures, fps, types_list)
    
    # Compute PCA coordinates
    print(f"Computing PCA with {n_pca_components} components...")
    pc_coordinates, explained_variance = compute_pca_coordinates(fp_diff_vectors, n_pca_components)
    
    print(f"PCA explained variance (top {min(5, len(explained_variance))} components): "
          f"{explained_variance[:min(5, len(explained_variance))]}")
    print(f"Total variance explained: {np.sum(explained_variance):.4f}")
    
    # Analyze PC coordinate distribution to set parameters
    pc_distances = []
    for i in range(n_structures):
        for j in range(i+1, n_structures):
            dist = np.linalg.norm(pc_coordinates[i] - pc_coordinates[j])
            pc_distances.append(dist)
    
    dist_array = np.array(pc_distances)
    
    # Calculate distribution statistics
    median_dist = np.median(dist_array)
    q25_dist = np.percentile(dist_array, 25)
    q75_dist = np.percentile(dist_array, 75)
    iqr = q75_dist - q25_dist
    
    # Adjust min_cluster_size based on variance and dataset size
    # For small datasets, use smaller cluster sizes
    if n_structures < 20:
        adjusted_min_cluster_size = 2
    elif n_structures < 50:
        adjusted_min_cluster_size = max(2, int(n_structures * 0.1))
    else:
        adjusted_min_cluster_size = max(
            min_cluster_size,
            int(n_structures * 0.05)
        )
        adjusted_min_cluster_size = min(adjusted_min_cluster_size, n_structures // 5)
    
    # Ensure min_cluster_size is at least 2 (HDBSCAN requirement)
    adjusted_min_cluster_size = max(2, adjusted_min_cluster_size)
    
    print(f"PC distance stats: median={median_dist:.4f}, q25={q25_dist:.4f}, q75={q75_dist:.4f}, IQR={iqr:.4f}")
    print(f"Adjusted min_cluster_size={adjusted_min_cluster_size}")
    
    # Perform HDBSCAN clustering in PCA space with Euclidean metric
    print(f"Performing HDBSCAN clustering in PCA space...")
    clusterer = HDBSCAN(min_cluster_size=adjusted_min_cluster_size, 
                       metric='euclidean')
    cluster_labels = clusterer.fit_predict(pc_coordinates)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters >= 0])  # Exclude noise (-1)
    
    print(f"Found {n_clusters} clusters and {np.sum(cluster_labels == -1)} noise points")
    
    # Calculate centroids for each cluster in PCA space
    centroids = []
    centroid_indices = []
    for cluster_id in unique_clusters:
        if cluster_id < 0:  # Skip noise points
            continue
            
        cluster_members = np.where(cluster_labels == cluster_id)[0]
        
        # Calculate centroid as the point with minimum sum of Euclidean distances
        cluster_pc_coords = pc_coordinates[cluster_members]
        dist_to_others = np.zeros(len(cluster_members))
        
        for i in range(len(cluster_members)):
            for j in range(len(cluster_members)):
                dist_to_others[i] += np.linalg.norm(cluster_pc_coords[i] - cluster_pc_coords[j])
        
        centroid_idx = np.argmin(dist_to_others)
        centroid_structure_idx = cluster_members[centroid_idx]
        
        centroids.append(cluster_id)
        centroid_indices.append(centroid_structure_idx)
    
    selected_indices = []
    rejected_indices = []
    # Track which centroid each selected structure belongs to
    centroid_memberships = {}  # {structure_index: centroid_cluster_id}
    selection_reasons = []  # For visualization
    
    # Process each cluster
    for cluster_id in unique_clusters:
        if cluster_id < 0:  # Skip noise points for now
            continue
            
        cluster_members = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_members)
        
        if cluster_size == 1:
            # If only one member, add it directly
            selected_indices.append(cluster_members[0])
            centroid_memberships[cluster_members[0]] = cluster_id
            selection_reasons.append('single')
            continue
            
        # Get PC coordinates for this cluster
        cluster_pc_coords = pc_coordinates[cluster_members]
        
        # Find the centroid (already computed)
        centroid_idx = centroid_indices[centroids.index(cluster_id)]
        centroid_local_idx = np.where(cluster_members == centroid_idx)[0][0]
        
        # Add centroid structure
        if centroid_idx not in selected_indices:
            selected_indices.append(centroid_idx)
            centroid_memberships[centroid_idx] = cluster_id
            selection_reasons.append('centroid')
        
            # Calculate distances in PCA space
            pc_dists_from_centroid = np.array([
                np.linalg.norm(cluster_pc_coords[i] - cluster_pc_coords[centroid_local_idx])
                for i in range(cluster_size)
            ])
            
            median_dist_from_centroid = np.median(pc_dists_from_centroid)
            
            # Calculate adaptive half-radius based on cluster size
            half_radius = median_dist_from_centroid * 0.5
            
            # For small clusters, increase the proportion
            if cluster_size < 10:
                half_radius = median_dist_from_centroid * 0.7
            # For large clusters, use a more conservative radius
            elif cluster_size > 30:
                half_radius = median_dist_from_centroid * 0.4
            
            print(f"Cluster {cluster_id}: size={cluster_size}, "
                  f"median_dist={median_dist_from_centroid:.4f}, half_radius={half_radius:.4f}")
            
            # Define the band around half-radius
            skin_depth = max(0.0001, half_radius * 0.15)
            lower_bound = half_radius - skin_depth
            upper_bound = half_radius + skin_depth
            
            print(f"Searching for structures in band: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            # Find structures within the band
            in_band = [(i, d) for i, d in enumerate(pc_dists_from_centroid) 
                      if lower_bound <= d <= upper_bound and i != centroid_local_idx]
            
            # Sort by how close they are to exactly half-radius
            in_band.sort(key=lambda x: abs(x[1] - half_radius))
            
            # Limit the number of structures to select from this cluster
            n_to_select = max(1, int(max_structures * (cluster_size / n_structures)))
            n_to_select = min(n_to_select, len(in_band) + 1)  # +1 for centroid
            
            print(f"Cluster {cluster_id}: Found {len(in_band)} structures in half-radius band, "
                  f"selecting up to {n_to_select-1} (already added centroid)")
            
            # Add structures from the band
            for i, _ in in_band[:n_to_select-1]:
                member_idx = cluster_members[i]
                if member_idx not in selected_indices:
                    selected_indices.append(member_idx)
                    centroid_memberships[member_idx] = cluster_id
                    selection_reasons.append('half_radius')
        
        # Track rejected structures for visualization
        for i in range(cluster_size):
            if cluster_members[i] not in selected_indices:
                rejected_indices.append(cluster_members[i])
    
    # Handle noise points (-1) separately
    noise_points = np.where(cluster_labels == -1)[0]
    if len(noise_points) > 0:
        # For noise points, select those most distant from already selected structures in PCA space
        if len(selected_indices) > 0:
            noise_distances = np.zeros((len(noise_points), len(selected_indices)))
            for i, noise_idx in enumerate(noise_points):
                for j, selected_idx in enumerate(selected_indices):
                    noise_distances[i, j] = np.linalg.norm(pc_coordinates[noise_idx] - pc_coordinates[selected_idx])
            
            # For each noise point, get its minimum distance to any selected structure
            min_distances = np.min(noise_distances, axis=1)
            
            # Sort noise points by their minimum distance (descending)
            sorted_noise = [(idx, dist) for idx, dist in zip(noise_points, min_distances)]
            sorted_noise.sort(key=lambda x: x[1], reverse=True)
            
            # Select some portion of noise points, prioritizing those most different
            n_noise_to_add = min(len(noise_points) // 4, max_structures - len(selected_indices))
            for i in range(min(n_noise_to_add, len(sorted_noise))):
                selected_indices.append(sorted_noise[i][0])
                centroid_memberships[sorted_noise[i][0]] = -1
                selection_reasons.append('noise')
            
            # Track rejected noise points
            for noise_idx in noise_points:
                if noise_idx not in selected_indices:
                    rejected_indices.append(noise_idx)
    
    # If we still need more structures, add more from rejected ones
    # based on maximizing distance from already selected structures in PCA space
    remaining = max_structures - len(selected_indices)
    if remaining > 0 and len(rejected_indices) > 0:
        candidate_distances = np.zeros((len(rejected_indices), len(selected_indices)))
        for i, rejected_idx in enumerate(rejected_indices):
            for j, selected_idx in enumerate(selected_indices):
                candidate_distances[i, j] = np.linalg.norm(pc_coordinates[rejected_idx] - pc_coordinates[selected_idx])
        
        # For each rejected structure, get its minimum distance to any selected structure
        min_distances = np.min(candidate_distances, axis=1)
        
        # Sort rejected structures by their minimum distance (descending)
        sorted_candidates = [(idx, dist) for idx, dist in zip(rejected_indices, min_distances)]
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add the most distant rejected structures
        for i in range(min(remaining, len(sorted_candidates))):
            selected_indices.append(sorted_candidates[i][0])
            # Assign these to the nearest centroid or -2 if none
            min_dist = float('inf')
            nearest_centroid = -2
            for cent_idx, cent_cluster in centroid_memberships.items():
                if selection_reasons[selected_indices.index(cent_idx)] == 'centroid':
                    dist = np.linalg.norm(pc_coordinates[sorted_candidates[i][0]] - pc_coordinates[cent_idx])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_centroid = cent_cluster
            centroid_memberships[sorted_candidates[i][0]] = nearest_centroid
            selection_reasons.append('additional')
    
    print(f"Selected {len(selected_indices)} structures: {selection_reasons.count('centroid')} centroids, "
          f"{selection_reasons.count('half_radius')} half-radius, {selection_reasons.count('noise')} noise, "
          f"{selection_reasons.count('additional')} additional")
    
    # Visualize results if requested
    if visualize:
        plt.figure(figsize=(12, 10))
        
        # Use first 2 PCs for visualization
        coords_2d = pc_coordinates[:, :2]
        
        # Plot all structures as background with cluster coloring
        for cluster_id in unique_clusters:
            # Find all structures in this cluster (including rejected ones)
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Plot rejected structures in this cluster as gray
            rejected_in_cluster = [idx for idx in cluster_indices if idx in rejected_indices]
            if rejected_in_cluster:
                plt.scatter(coords_2d[rejected_in_cluster, 0], coords_2d[rejected_in_cluster, 1], 
                           c='lightgray', alpha=0.3, s=30, marker='o')
        
        # Generate colors for centroids - one distinct color per cluster
        from matplotlib.colors import to_rgba
        
        # Get unique centroid cluster IDs (excluding noise which is -1)
        unique_centroid_clusters = set(v for v in centroid_memberships.values() if v >= 0)
        
        # Create color map with distinct colors for each cluster
        if hasattr(mpl.cm, 'colormaps'):  # Matplotlib 3.7+
            color_map = mpl.colormaps['tab10']
        elif hasattr(mpl, 'colormaps'):  # Alternative syntax for newer versions
            color_map = mpl.colormaps['tab10']
        else:  # Fall back to older version for compatibility
            color_map = cm.get_cmap('tab10', max(10, len(unique_centroid_clusters)))
            
        # Plot centroids first
        for cluster_id in unique_centroid_clusters:
            # Find centroid for this cluster
            centroid_idx_list = [idx for idx, reason in zip(selected_indices, selection_reasons) 
                               if reason == 'centroid' and centroid_memberships[idx] == cluster_id]
            
            if centroid_idx_list:
                centroid_color = color_map(cluster_id % 10)  # Get distinct color
                plt.scatter(coords_2d[centroid_idx_list, 0], coords_2d[centroid_idx_list, 1],
                           label=f'Centroid {cluster_id}', color=centroid_color,
                           marker='*', s=200, edgecolor='black', alpha=1.0)
                
                # Find half-radius points associated with this centroid
                half_radius_indices = [idx for idx, reason in zip(selected_indices, selection_reasons)
                                      if reason == 'half_radius' and centroid_memberships[idx] == cluster_id]
                
                if half_radius_indices:
                    # Use a lighter version of the same color for half-radius points
                    half_radius_color = to_rgba(centroid_color, alpha=0.7)
                    plt.scatter(coords_2d[half_radius_indices, 0], coords_2d[half_radius_indices, 1],
                               label=f'Half-radius {cluster_id}', color=half_radius_color,
                               marker='o', s=100, edgecolor='black', alpha=0.7)
        
        # Plot single-member clusters
        single_indices = [idx for idx, reason in zip(selected_indices, selection_reasons) if reason == 'single']
        if single_indices:
            plt.scatter(coords_2d[single_indices, 0], coords_2d[single_indices, 1],
                       label='Single member', color='purple', marker='s', s=100, alpha=0.8)
        
        # Plot noise points
        noise_indices = [idx for idx, reason in zip(selected_indices, selection_reasons) if reason == 'noise']
        if noise_indices:
            plt.scatter(coords_2d[noise_indices, 0], coords_2d[noise_indices, 1],
                       label='Noise', color='green', marker='d', s=100, alpha=0.8)
        
        # Plot additional points
        additional_indices = [idx for idx, reason in zip(selected_indices, selection_reasons) if reason == 'additional']
        if additional_indices:
            plt.scatter(coords_2d[additional_indices, 0], coords_2d[additional_indices, 1],
                       label='Additional', color='orange', marker='p', s=100, alpha=0.8)
        
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.title(f'Selected Structures for {n_substitutions} Substitutions (PCA Space)')
        
        plt.tight_layout()
        plt.savefig(f'selected_{n_substitutions}_substitutions.png', dpi=300, bbox_inches='tight')
    
    return sorted(selected_indices)

def POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, max_iter=5000, 
                       visualize=False, n_pca_components=10, kim_model=None, energy_percentile=80):
    """
    Generate diverse structures using random substitutions followed by PCA-based clustering
    and selection based on centroids and half-radius points.
    
    Parameters:
    - atoms_origin: Original ASE Atoms object
    - elem_from: Element to substitute
    - elem_to: New element
    - max_subs: Maximum number of substitutions
    - max_structures: Maximum number of structures per substitution level
    - max_iter: Maximum number of random substitutions to try
    - visualize: Whether to generate visualization plots
    - n_pca_components: Number of PCA components for clustering
    - kim_model: KIM model name for energy filtering (None to disable)
    - energy_percentile: Energy percentile threshold for filtering
    
    Returns:
    - List of selected structures
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
        print(f"Generating structures with {n_C} substitution(s)...")
        
        # Track unique substitution patterns to avoid duplicates
        generated_patterns = set()
        
        # Store structures for this substitution level
        structures_n_C = []
        fps_n_C = []
        types_n_C = []
        
        # For single substitution, generate all possible structures
        if n_C == 1:
            for idx in from_indices:
                new_atoms = atoms_origin.copy()
                new_atoms[idx].symbol = elem_to
                
                # Sort atoms
                sorted_atoms = sort(new_atoms)
                
                # Calculate fingerprint
                fp = get_fp_mat(atoms=sorted_atoms)
                types = np.int32(read_types(sorted_atoms))
                
                # Store the structure
                structures_n_C.append(sorted_atoms)
                fps_n_C.append(fp)
                types_n_C.append(types)
                
            print(f"Generated {len(structures_n_C)} structures with {n_C} substitution (all possible configurations)")
            # No special handling for n_C=1 anymore - we always apply clustering to find symmetrically equivalent structures
        else:
            # Generate structures with n_C substitutions
            for i in range(max_iter):
                if len(structures_n_C) >= max_structures * 3:  # Generate 3x more for better diversity
                    break
                    
                new_atoms = atoms_origin.copy()

                # Randomly select atoms to substitute
                subs_indices = tuple(sorted(sample(from_indices, n_C)))
                
                # Skip if we've already tried this pattern
                if subs_indices in generated_patterns:
                    continue
                    
                generated_patterns.add(subs_indices)
                
                # Apply substitution
                for idx in subs_indices:
                    new_atoms[idx].symbol = elem_to

                # Sort atoms
                sorted_atoms = sort(new_atoms)
                
                # Calculate fingerprint
                fp = get_fp_mat(atoms=sorted_atoms)
                types = np.int32(read_types(sorted_atoms))
                
                # Store the structure
                structures_n_C.append(sorted_atoms)
                fps_n_C.append(fp)
                types_n_C.append(types)
                
            print(f"Generated {len(structures_n_C)} candidate structures with {n_C} substitution(s)")
        
        if len(structures_n_C) > 0:
            # Select diverse structures using PCA-based clustering
            min_cluster_size = 10
            if n_C == 1 and len(structures_n_C) < 10:
                # For very few structures with single substitution, adjust parameters
                min_cluster_size = 5
            
            selected_indices = select_diverse_structures(
                structures_n_C, fps_n_C, types_n_C, 
                max_structures=max_structures,
                min_cluster_size=min_cluster_size,
                n_pca_components=n_pca_components,
                visualize=visualize,
                n_substitutions=n_C
            )
            
            print(f"Selected {len(selected_indices)} diverse structures out of {len(structures_n_C)}")
            
            # Apply KIM energy filtering if requested
            if kim_model is not None and len(selected_indices) > 1:
                print(f"Filtering selected structures by KIM energy...")
                selected_structures = [structures_n_C[idx] for idx in selected_indices]
                filtered_indices, energies = filter_by_kim_energy(
                    selected_structures,
                    kim_model, 
                    energy_percentile
                )
                # Map back to original indices
                selected_indices = [selected_indices[i] for i in filtered_indices]
                print(f"After energy filtering: {len(selected_indices)} structures remain")
            
            # Write selected structures to POSCAR files
            for i, idx in enumerate(selected_indices):
                selected_struct = structures_n_C[idx]
                poscar_name = f'POSCAR_{n_C}_{i+1}'
                ase.io.write(poscar_name, selected_struct, format='vasp', direct=True, vasp5=True)
                print(f"Wrote structure to: {poscar_name}")
                final_structures.append(selected_struct)
                
    print(f"POSCAR_GEN_CLUSTER: Finished generating {len(final_structures)} diverse structures.")
    return final_structures

def POST_PROC(caldir):
    """
    Post-processing function - with the new approach, less work is needed here
    since filtering is done during structure generation.
    """
    print("Post-processing complete. Diverse structures have been generated.")
    # No need to call AFLOW for filtering as it's handled by our clustering approach

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
    print("Enter number of PCA components (default 10): ", end="", flush=True)
    n_pca_input = input().strip()
    n_pca_components = int(n_pca_input) if n_pca_input else 10
    print("Use KIM energy filtering? (y/n): ", end="", flush=True)
    use_kim = input().lower() in ['y', 'yes']
    kim_model = "Tersoff_LAMMPS_Tersoff_1989_SiGe__MO_350526375143_004" if use_kim else None
    print("Generate visualization plots? (y/n): ", end="", flush=True)
    vis_input = input().lower()
    visualize = vis_input == 'y' or vis_input == 'yes'
    
    if visualize:
        print("Visualization enabled - plots will be generated")
    
    if kim_model:
        print(f"KIM energy filtering enabled using model: {kim_model}")
    
    # Generate structures using the new PCA-based clustering approach
    POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, 
                       visualize=visualize, n_pca_components=n_pca_components, 
                       kim_model=kim_model)

    # Call post-processing function
    print("All substitutions complete. Starting post-processing...")
    POST_PROC(caldir)