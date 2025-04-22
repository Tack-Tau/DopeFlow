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

def pairwise_fp_dist(structures, fps, types_list, reference_fp=None, reference_types=None):
    """
    Compute pairwise fingerprint distances between structures.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - fps: List of fingerprint matrices
    - types_list: List of type arrays
    - reference_fp: Optional reference fingerprint (e.g., original POSCAR)
    - reference_types: Optional reference types (e.g., original POSCAR)
    
    Returns:
    - Distance matrix
    """
    n_struct = len(structures)
    dist_matrix = np.zeros((n_struct, n_struct))
    
    # If reference is provided, compute distances relative to reference
    if reference_fp is not None and reference_types is not None:
        for i in range(n_struct):
            # Distance from each structure to reference
            ref_dist = get_fp_dist(fps[i], reference_fp, types_list[i])
            
            # Store reference distance on diagonal
            dist_matrix[i, i] = ref_dist
            
            for j in range(i+1, n_struct):
                # Calculate distance between structures i and j
                struct_dist = get_fp_dist(fps[i], fps[j], types_list[i])
                
                # Store the distances
                dist_matrix[i, j] = struct_dist
                dist_matrix[j, i] = struct_dist
    else:
        # Original pairwise distance calculation
        for i in range(n_struct):
            for j in range(i+1, n_struct):
                dist = get_fp_dist(fps[i], fps[j], types_list[i])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
            
    return dist_matrix

def compute_structure_features(structures, fps):
    """
    Extract features from fingerprint matrices for clustering and PCA.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - fps: List of fingerprint matrices
    
    Returns:
    - Feature matrix (n_structures, n_features)
    """
    n_struct = len(structures)
    
    # For simplicity, we'll flatten and concatenate fingerprints
    # More sophisticated feature extraction can be implemented if needed
    features = []
    
    for i in range(n_struct):
        # Get the flattened fingerprint as features
        fp_flat = fps[i].flatten()
        # Take a subset if dimensions are too large
        if len(fp_flat) > 1000:
            indices = np.linspace(0, len(fp_flat)-1, 1000, dtype=int)
            fp_flat = fp_flat[indices]
        features.append(fp_flat)
    
    # Handle variable-length features
    max_len = max(len(feat) for feat in features)
    padded_features = np.zeros((n_struct, max_len))
    
    for i, feat in enumerate(features):
        padded_features[i, :len(feat)] = feat
        
    return padded_features

def select_diverse_structures(structures, fps, types_list, max_structures=10, min_cluster_size=2, eps=1e-4, 
                             visualize=False, n_substitutions=None, reference_fp=None, reference_types=None,
                             log_scale=True, merge_factor=1.2):
    """
    Select diverse structures using HDBSCAN clustering and selecting structures at 
    centroids and half-radius points.
    
    Parameters:
    - structures: List of ASE Atoms objects
    - fps: List of fingerprint matrices
    - types_list: List of type arrays
    - max_structures: Maximum number of structures to select
    - min_cluster_size: Minimum size of a cluster in HDBSCAN
    - eps: Tolerance band around half-radius (skin depth)
    - visualize: Whether to generate visualization plots
    - n_substitutions: Number of substitutions for labeling plots
    - reference_fp: Optional reference fingerprint (e.g., original POSCAR)
    - reference_types: Optional reference types (e.g., original POSCAR)
    - log_scale: Whether to use log scale for distance matrix and visualization
    - merge_factor: Factor for merging clusters (lower values preserve more clusters)
    
    Returns:
    - List of indices of selected structures
    """
    n_structures = len(structures)
    
    if n_structures <= max_structures:
        # If we have fewer structures than requested, return all
        return list(range(n_structures))
    
    # Compute distance matrix using fingerprint distance
    print(f"Computing distance matrix for {n_structures} structures...")
    original_distance_matrix = pairwise_fp_dist(structures, fps, types_list, reference_fp, reference_types)
    
    # Keep a copy of the original matrix for half-radius calculations
    distance_matrix = original_distance_matrix.copy()
    
    # Apply log scale if requested (common in materials science for better handling of different scales)
    # Avoid log(0) by adding a small offset
    if log_scale:
        min_nonzero = np.min(distance_matrix[distance_matrix > 0])
        offset = min_nonzero * 0.1  # Small offset to avoid log(0)
        transformed_distance_matrix = np.log10(distance_matrix + offset)
        # Ensure positivity by shifting if needed
        if np.min(transformed_distance_matrix) < 0:
            transformed_distance_matrix -= np.min(transformed_distance_matrix)
        clustering_distance_matrix = transformed_distance_matrix
        print(f"Using log-scaled distances for clustering")
    else:
        clustering_distance_matrix = distance_matrix
    
    # Analyze distance distribution to set parameters dynamically
    flat_distances = []
    for i in range(n_structures):
        for j in range(i+1, n_structures):
            flat_distances.append(clustering_distance_matrix[i, j])
    
    dist_array = np.array(flat_distances)
    
    # Calculate distribution statistics
    median_dist = np.median(dist_array)
    q25_dist = np.percentile(dist_array, 25)  # 25th percentile
    q75_dist = np.percentile(dist_array, 75)  # 75th percentile
    iqr = q75_dist - q25_dist  # Interquartile range
    
    # 1. Adjust min_cluster_size based on data distribution and size
    # Smaller IQR = more homogeneous data = larger min_cluster_size
    adjusted_min_cluster_size = max(
        min_cluster_size, 
        int(n_structures * (0.05 + 0.1 * np.exp(-iqr)))
    )
    adjusted_min_cluster_size = min(adjusted_min_cluster_size, n_structures // 5)
    
    # 2. Set epsilon based on distance distribution
    # Use interquartile range to determine tolerance
    epsilon = q25_dist * (0.3 + 0.4 * (iqr/median_dist))
    
    print(f"Distance stats: median={median_dist:.4f}, q25={q25_dist:.4f}, q75={q75_dist:.4f}, IQR={iqr:.4f}")
    print(f"Adjusted parameters: min_cluster_size={adjusted_min_cluster_size}, epsilon={epsilon:.4f}")
    
    # Perform HDBSCAN clustering with adjusted parameters
    print(f"Performing HDBSCAN clustering with adaptive parameters...")
    clusterer = HDBSCAN(min_cluster_size=adjusted_min_cluster_size, 
                       metric='precomputed',
                       cluster_selection_epsilon=epsilon)
    cluster_labels = clusterer.fit_predict(clustering_distance_matrix)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters >= 0])  # Exclude noise (-1)
    
    print(f"Found {n_clusters} clusters and {np.sum(cluster_labels == -1)} noise points")
    
    # Prepare for visualization if requested
    if visualize:
        # Use MDS for 2D embedding of distance matrix
        from sklearn.manifold import MDS
        embedding = MDS(n_components=2, dissimilarity='precomputed', 
                       random_state=42, normalized_stress='auto')
        coords_2d = embedding.fit_transform(clustering_distance_matrix)
    
    # Calculate centroids for each cluster
    centroids = []
    centroid_indices = []
    for cluster_id in unique_clusters:
        if cluster_id < 0:  # Skip noise points
            continue
            
        cluster_members = np.where(cluster_labels == cluster_id)[0]
        
        # Create distance matrix for just this cluster
        cluster_size = len(cluster_members)
        cluster_distances = np.zeros((cluster_size, cluster_size))
        for i, idx_i in enumerate(cluster_members):
            for j, idx_j in enumerate(cluster_members):
                cluster_distances[i, j] = clustering_distance_matrix[idx_i, idx_j]
        
        # Calculate distance to centroid for each structure in the cluster
        # Centroid is the point with minimum sum of distances to all other points
        dist_to_others = cluster_distances.sum(axis=1)
        centroid_idx = np.argmin(dist_to_others)
        centroid_structure_idx = cluster_members[centroid_idx]
        
        centroids.append(cluster_id)
        centroid_indices.append(centroid_structure_idx)
    
    # Check if centroids are well-separated, merge clusters if needed
    # This ensures centroids are at least double their half-radius apart
    merged_clusters = {}  # Maps original cluster ID to merged cluster ID
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            ci_idx = centroid_indices[i]
            cj_idx = centroid_indices[j]
            
            # Distance between centroids
            centroid_dist = clustering_distance_matrix[ci_idx, cj_idx]
            
            # Get the cluster members for both clusters
            ci_members = np.where(cluster_labels == centroids[i])[0]
            cj_members = np.where(cluster_labels == centroids[j])[0]
            
            # Calculate density metrics for clusters using adaptive weighting based on AFLOW approach
            ci_median_dist = np.median([clustering_distance_matrix[ci_idx, m] for m in ci_members])
            cj_median_dist = np.median([clustering_distance_matrix[cj_idx, m] for m in cj_members])
            
            # Calculate adaptive half-radius based on cluster density and AFLOW inspiration
            # Use median distance rather than direct density calculation
            ci_radius = ci_median_dist * 0.5  # Half the median distance to members
            cj_radius = cj_median_dist * 0.5  # Half the median distance to members
            
            # Check if centroids are too close relative to their radii
            min_required_dist = merge_factor * (ci_radius + cj_radius)
            
            if centroid_dist < min_required_dist:
                print(f"Merging clusters {centroids[i]} and {centroids[j]} - too close: "
                      f"dist={centroid_dist:.4f}, required={min_required_dist:.4f}")
                
                # Determine which cluster to keep (usually the one with more members)
                keep_cluster = centroids[i] if len(ci_members) >= len(cj_members) else centroids[j]
                merge_cluster = centroids[j] if len(ci_members) >= len(cj_members) else centroids[i]
                
                # Update the mapping for the merged cluster
                merged_clusters[merge_cluster] = keep_cluster
    
    # Apply cluster merging to all points
    if merged_clusters:
        for i in range(len(cluster_labels)):
            if cluster_labels[i] in merged_clusters:
                cluster_labels[i] = merged_clusters[cluster_labels[i]]
        
        # Recalculate unique clusters and centroids after merging
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters[unique_clusters >= 0])
        print(f"After merging: {n_clusters} clusters and {np.sum(cluster_labels == -1)} noise points")
        
        # Ensure cluster IDs start from 1 (except noise points which are -1)
        # This addresses the issue where merged clusters always start from 0
        new_cluster_mapping = {}
        next_id = 1
        for cluster_id in unique_clusters:
            if cluster_id >= 0:  # Skip noise points
                new_cluster_mapping[cluster_id] = next_id
                next_id += 1
        
        # Apply the new mapping
        for i in range(len(cluster_labels)):
            if cluster_labels[i] >= 0:  # Skip noise points
                cluster_labels[i] = new_cluster_mapping[cluster_labels[i]]
        
        # Update unique_clusters with the new mapping
        unique_clusters = np.unique(cluster_labels)
        
        # Recalculate centroids with updated cluster IDs
        centroids = []
        centroid_indices = []
        for cluster_id in unique_clusters:
            if cluster_id < 0:  # Skip noise points
                continue
                
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            # Create distance matrix for just this cluster
            cluster_size = len(cluster_members)
            cluster_distances = np.zeros((cluster_size, cluster_size))
            for i, idx_i in enumerate(cluster_members):
                for j, idx_j in enumerate(cluster_members):
                    cluster_distances[i, j] = clustering_distance_matrix[idx_i, idx_j]
            
            # Calculate distance to centroid for each structure in the cluster
            dist_to_others = cluster_distances.sum(axis=1)
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
            
        # Create distance matrices for just this cluster - both original and clustering versions
        cluster_distances_original = np.zeros((cluster_size, cluster_size))
        cluster_distances_clustering = np.zeros((cluster_size, cluster_size))
        
        for i, idx_i in enumerate(cluster_members):
            for j, idx_j in enumerate(cluster_members):
                cluster_distances_original[i, j] = original_distance_matrix[idx_i, idx_j]
                cluster_distances_clustering[i, j] = clustering_distance_matrix[idx_i, idx_j]
        
        # Use clustering distance matrix to find the centroid
        dist_to_others = cluster_distances_clustering.sum(axis=1)
        centroid_idx = np.argmin(dist_to_others)
        centroid_structure_idx = cluster_members[centroid_idx]
        
        # Add centroid structure
        if centroid_structure_idx not in selected_indices:
            selected_indices.append(centroid_structure_idx)
            centroid_memberships[centroid_structure_idx] = cluster_id
            selection_reasons.append('centroid')
        
            # Calculate median distance using the ORIGINAL distance matrix
            # This ensures we select structures that are properly half-radius in the original space
            original_dists_from_centroid = cluster_distances_original[centroid_idx]
            median_dist_from_centroid = np.median(original_dists_from_centroid)
            
            # Calculate adaptive half-radius based on cluster profile using original distances
            half_radius = median_dist_from_centroid * 0.5
            
            # For small clusters, increase the proportion of the half-radius
            if cluster_size < 10:
                half_radius = median_dist_from_centroid * 0.7
            
            # For large clusters, use a more conservative radius
            elif cluster_size > 30:
                half_radius = median_dist_from_centroid * 0.4
            
            print(f"Cluster {cluster_id}: size={cluster_size}, "
                  f"median_dist={median_dist_from_centroid:.4f}, half_radius={half_radius:.4f}")
            
            # Find structures at approximately half-radius in ORIGINAL distance space
            dist_from_centroid = original_dists_from_centroid
            
            # Define the band around half-radius
            # Use an adaptive skin depth based on half-radius value
            skin_depth = max(eps, half_radius * 0.15)  # Increased from 0.1 to 0.15 for wider band
            lower_bound = half_radius - skin_depth
            upper_bound = half_radius + skin_depth
            
            print(f"Searching for structures in band: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            # Find structures within the band
            in_band = [(i, d) for i, d in enumerate(dist_from_centroid) 
                      if lower_bound <= d <= upper_bound and i != centroid_idx]
            
            # Sort by how close they are to exactly half-radius
            in_band.sort(key=lambda x: abs(x[1] - half_radius))
            
            # Limit the number of structures to select from this cluster
            # More structures from larger clusters
            n_to_select = max(1, int(max_structures * (cluster_size / n_structures)))
            n_to_select = min(n_to_select, len(in_band) + 1)  # +1 for centroid (already added)
            
            print(f"Cluster {cluster_id}: Found {len(in_band)} structures in half-radius band, "
                  f"selecting up to {n_to_select-1} (already added centroid)")
            
            # Add structures from the band, limited by n_to_select
            # We've already added the centroid, so select n_to_select-1 more
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
        # For noise points, select those most distant from already selected structures
        # This ensures we capture outliers that might be interesting
        if len(selected_indices) > 0:
            noise_distances = np.zeros((len(noise_points), len(selected_indices)))
            for i, noise_idx in enumerate(noise_points):
                for j, selected_idx in enumerate(selected_indices):
                    noise_distances[i, j] = clustering_distance_matrix[noise_idx, selected_idx]
            
            # For each noise point, get its minimum distance to any selected structure
            min_distances = np.min(noise_distances, axis=1)
            
            # Sort noise points by their minimum distance (descending)
            sorted_noise = [(idx, dist) for idx, dist in zip(noise_points, min_distances)]
            sorted_noise.sort(key=lambda x: x[1], reverse=True)
            
            # Select some portion of noise points, prioritizing those most different
            n_noise_to_add = min(len(noise_points) // 4, max_structures - len(selected_indices))
            for i in range(min(n_noise_to_add, len(sorted_noise))):
                selected_indices.append(sorted_noise[i][0])
                centroid_memberships[sorted_noise[i][0]] = -1  # Noise points don't belong to any centroid
                selection_reasons.append('noise')
            
            # Track rejected noise points
            for noise_idx in noise_points:
                if noise_idx not in selected_indices:
                    rejected_indices.append(noise_idx)
    
    # If we still need more structures, add more from rejected ones
    # based on maximizing distance from already selected structures
    remaining = max_structures - len(selected_indices)
    if remaining > 0 and len(rejected_indices) > 0:
        candidate_distances = np.zeros((len(rejected_indices), len(selected_indices)))
        for i, rejected_idx in enumerate(rejected_indices):
            for j, selected_idx in enumerate(selected_indices):
                candidate_distances[i, j] = clustering_distance_matrix[rejected_idx, selected_idx]
        
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
                    dist = clustering_distance_matrix[sorted_candidates[i][0], cent_idx]
                    if dist < min_dist:
                        min_dist = dist
                        nearest_centroid = cent_cluster
            centroid_memberships[sorted_candidates[i][0]] = nearest_centroid
            selection_reasons.append('additional')
    
    print(f"Selected {len(selected_indices)} structures: {selection_reasons.count('centroid')} centroids, "
          f"{selection_reasons.count('half_radius')} half-radius, {selection_reasons.count('noise')} noise, "
          f"{selection_reasons.count('additional')} additional")
    
    # Visualize results if requested
    if visualize and 'coords_2d' in locals():
        plt.figure(figsize=(12, 10))
        
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
        # Fix for matplotlib 3.7+ deprecation warning
        if hasattr(mpl.cm, 'colormaps'):  # Matplotlib 3.7+
            color_map = mpl.colormaps['tab10']
        elif hasattr(mpl, 'colormaps'):  # Alternative syntax for newer versions
            color_map = mpl.colormaps['tab10']
        else:  # Fall back to older version for compatibility
            color_map = cm.get_cmap('tab10', max(10, len(unique_centroid_clusters)))
            
        # Plot centroids first
        for cluster_id in unique_centroid_clusters:
            # Find centroid for this cluster
            centroid_indices = [idx for idx, reason in zip(selected_indices, selection_reasons) 
                               if reason == 'centroid' and centroid_memberships[idx] == cluster_id]
            
            if centroid_indices:
                centroid_color = color_map(cluster_id % 10)  # Get distinct color
                plt.scatter(coords_2d[centroid_indices, 0], coords_2d[centroid_indices, 1],
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
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        if log_scale:
            plt.title(f'Selected Structures for {n_substitutions} Substitutions (Log Scale)')
        else:
            plt.title(f'Selected Structures for {n_substitutions} Substitutions')
        
        plt.tight_layout()
        plt.savefig(f'selected_{n_substitutions}_substitutions.png', dpi=300, bbox_inches='tight')
    
    return sorted(selected_indices)

def POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, max_iter=5000, visualize=False):
    """
    Generate diverse structures using random substitutions followed by clustering and selection
    based on centroids and half-radius points.
    
    Parameters:
    - atoms_origin: Original ASE Atoms object
    - elem_from: Element to substitute
    - elem_to: New element
    - max_subs: Maximum number of substitutions
    - max_structures: Maximum number of structures per substitution level
    - max_iter: Maximum number of random substitutions to try
    - visualize: Whether to generate visualization plots
    
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

    # Precompute reference fingerprint from the original structure
    # This will be used as a reference for fingerprint distance calculations
    sorted_atoms_origin = sort(atoms_origin)
    reference_fp = get_fp_mat(atoms=sorted_atoms_origin)
    reference_types = np.int32(read_types(sorted_atoms_origin))
    print(f"Computed reference fingerprint from original POSCAR")

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
            # Select diverse structures using the same approach for all substitution levels
            min_cluster_size = 10
            if n_C == 1 and len(structures_n_C) < 10:
                # For very few structures with single substitution, adjust parameters
                min_cluster_size = 5
            
            selected_indices = select_diverse_structures(
                structures_n_C, fps_n_C, types_n_C, 
                max_structures=max_structures,
                min_cluster_size=min_cluster_size,
                eps=1e-4,  # Use much smaller skin-depth for more precise selection
                visualize=visualize,
                n_substitutions=n_C,
                reference_fp=reference_fp,  # Pass the reference fingerprint
                reference_types=reference_types,  # Pass the reference types
                log_scale=True,  # Use log scale as requested
                merge_factor=1.2  # Reduced from 1.5 to preserve more clusters
            )
            
            print(f"Selected {len(selected_indices)} diverse structures out of {len(structures_n_C)}")
            
            # Write selected structures to POSCAR files
            for i, idx in enumerate(selected_indices):
                selected_struct = structures_n_C[idx]
                poscar_name = f'POSCAR_{n_C}_{i+1}'
                ase.io.write(poscar_name, selected_struct, 'vasp', direct=True, long_format=True, vasp5=True)
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
    print("Generate visualization plots? (y/n): ", end="", flush=True)
    vis_input = input().lower()
    visualize = vis_input == 'y' or vis_input == 'yes'
    
    if visualize:
        print("Visualization enabled - plots will be generated")
    
    # Generate structures using the new clustering approach
    POSCAR_GEN_CLUSTER(atoms_origin, elem_from, elem_to, max_subs, max_structures, visualize=visualize)

    # Call post-processing function
    print("All substitutions complete. Starting post-processing...")
    POST_PROC(caldir)