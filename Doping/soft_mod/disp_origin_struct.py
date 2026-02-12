#!/Users/tonyspc/miniforge3/envs/pyxtal/bin/python3
"""
Displace structure along soft mode eigenvectors.
Creates POSCAR files with displacements from 0.01 to 0.1 Angstrom.
"""
import numpy as np
from phonopy import load
from phonopy.structure.atoms import PhonopyAtoms
try:
    from pymatgen.io.phonopy import get_phonopy_structure
    from pymatgen.symmetry.bandstructure import HighSymmKpath
except ImportError as e:
    print("="*70)
    print("ERROR: Pymatgen is required but not installed")
    print("="*70)
    print("\nThis script requires pymatgen for q-point selection.")
    print("\nInstall it using:")
    print("  conda install -c conda-forge pymatgen")
    print("  or")
    print("  pip install pymatgen")
    print("\n" + "="*70)
    import sys
    sys.exit(1)

def filter_small_components(eigvec, threshold=1e-6):
    """Filter out eigenvector components smaller than threshold."""
    # Take real part of eigenvector (eigenvectors are complex)
    filtered = np.real(eigvec.copy())
    mask = np.abs(filtered) < threshold
    filtered[mask] = 0.0
    return filtered

def normalize_displacement(eigvec, masses):
    """
    Convert mass-weighted eigenvector to displacement vector and normalize.
    
    Phonopy eigenvectors are mass-weighted. To get actual displacements:
    displacement_j = eigvec_j / sqrt(mass_j)
    
    Then normalize so the largest atomic displacement equals 1.0 Angstrom.
    """
    n_atoms = len(masses)
    # Reshape eigenvector to (n_atoms, 3)
    eigvec_reshaped = eigvec.reshape(n_atoms, 3)
    
    # Convert to real displacements by dividing by sqrt(mass)
    displacements = np.zeros_like(eigvec_reshaped)
    for i in range(n_atoms):
        displacements[i] = eigvec_reshaped[i] / np.sqrt(masses[i])
    
    # Find maximum atomic displacement magnitude
    atom_disp_magnitudes = np.linalg.norm(displacements, axis=1)
    max_disp = np.max(atom_disp_magnitudes)
    
    if max_disp > 0:
        # Normalize so max displacement is 1.0 Angstrom
        displacements = displacements / max_disp
    
    return displacements

def write_poscar(cell, positions, filename):
    """Write structure to POSCAR file."""
    with open(filename, 'w') as f:
        # Comment line
        symbols = cell.symbols
        unique_symbols = []
        for s in symbols:
            if s not in unique_symbols:
                unique_symbols.append(s)
        
        symbol_counts = {s: symbols.count(s) for s in unique_symbols}
        comment = ' '.join([f"{s}{symbol_counts[s]}" for s in unique_symbols])
        f.write(f"{comment}\n")
        
        # Scaling factor
        f.write("1.0\n")
        
        # Lattice vectors
        lattice = cell.cell
        for vec in lattice:
            f.write(f"  {vec[0]:20.16f} {vec[1]:20.16f} {vec[2]:20.16f}\n")
        
        # Element symbols
        f.write(' '.join(unique_symbols) + '\n')
        
        # Element counts
        f.write(' '.join([str(symbol_counts[s]) for s in unique_symbols]) + '\n')
        
        # Direct coordinates
        f.write("Direct\n")
        
        for i, pos in enumerate(positions):
            symbol = symbols[i]
            # Ensure positions are real numbers
            x, y, z = np.real(pos[0]), np.real(pos[1]), np.real(pos[2])
            f.write(f"  {x:20.16f} {y:20.16f} {z:20.16f} {symbol}\n")

def get_high_symm_kpoints(phonon):
    """Get high symmetry k-points from pymatgen."""
    try:
        from pymatgen.core import Structure, Lattice
        
        # Convert phonopy structure to pymatgen
        cell = phonon.primitive.cell
        positions = phonon.primitive.scaled_positions
        symbols = phonon.primitive.symbols
        
        lattice = Lattice(cell)
        structure = Structure(lattice, symbols, positions)
        
        # Get high symmetry k-path
        kpath = HighSymmKpath(structure)
        return kpath.kpath
    except Exception as e:
        print(f"Warning: Could not get k-path from pymatgen: {e}")
        return None

def choose_qpoint(phonon):
    """Interactive q-point selection."""
    print("\n" + "="*70)
    print("Q-POINT SELECTION")
    print("="*70)
    
    # Get high symmetry points
    kpath_info = get_high_symm_kpoints(phonon)
    
    if kpath_info and 'kpoints' in kpath_info:
        print("\nHigh-symmetry q-points available:")
        print(f"{'Label':>10} {'Fractional Coordinates':>30}")
        print("-" * 70)
        
        kpoints = kpath_info['kpoints']
        labels = sorted(kpoints.keys())
        
        for i, label in enumerate(labels, 1):
            coords = kpoints[label]
            print(f"{i:3d}. {label:>6} {coords[0]:10.6f} {coords[1]:10.6f} {coords[2]:10.6f}")
        
        print("\n" + "-" * 70)
        print("Options:")
        print("  - Enter number (1-{}) to select from list".format(len(labels)))
        print("  - Enter 'custom' to input custom coordinates")
        print("  - Press Enter for Gamma point [0, 0, 0]")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '':
            q_point = [0.0, 0.0, 0.0]
            q_name = 'Gamma'
        elif choice.lower() == 'custom':
            print("\nEnter q-point in fractional coordinates (e.g., 0.333 0.333 0.0):")
            coords_str = input("q-point: ").strip()
            coords = [float(x) for x in coords_str.split()]
            if len(coords) != 3:
                print("Error: Must provide 3 coordinates. Using Gamma point.")
                q_point = [0.0, 0.0, 0.0]
                q_name = 'Gamma'
            else:
                q_point = coords
                q_name = 'custom'
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(labels):
                    q_name = labels[idx]
                    q_point = list(kpoints[q_name])
                else:
                    print("Invalid selection. Using Gamma point.")
                    q_point = [0.0, 0.0, 0.0]
                    q_name = 'Gamma'
            except ValueError:
                print("Invalid input. Using Gamma point.")
                q_point = [0.0, 0.0, 0.0]
                q_name = 'Gamma'
    else:
        print("\nWarning: Could not detect high-symmetry k-path.")
        print("This may indicate an unusual structure.")
        print("Using Gamma point [0, 0, 0] by default.")
        q_point = [0.0, 0.0, 0.0]
        q_name = 'Gamma'
    
    return q_point, q_name

def main():
    print("Loading phonopy data...")
    phonon = load("phonopy_params.yaml")
    
    # Get the primitive cell and masses
    primitive = phonon.primitive
    masses = primitive.masses
    n_atoms = len(masses)
    
    print(f"Structure: {n_atoms} atoms")
    print(f"Chemical symbols: {primitive.symbols}")
    
    # Interactive q-point selection
    q_point, q_name = choose_qpoint(phonon)
    
    print(f"\n{'='*70}")
    print(f"Calculating phonons at {q_name} point: {q_point}")
    print(f"{'='*70}")
    
    phonon.run_qpoints([q_point], with_eigenvectors=True)
    qdict = phonon.get_qpoints_dict()
    
    frequencies = qdict["frequencies"][0]
    eigenvectors = qdict["eigenvectors"][0]
    
    print(f"\nFrequencies (THz):")
    for i in range(min(6, len(frequencies))):
        print(f"  Mode {i+1}: {frequencies[i]:10.6f} THz")
    
    # Get the two lowest frequency modes
    mode_indices = [0, 1]
    
    print(f"\nProcessing soft modes at {q_name} point...")
    
    # Warning for non-Gamma points
    if not np.allclose(q_point, [0, 0, 0]):
        print(f"\n{'!'*70}")
        print("WARNING: Non-Gamma point selected!")
        print(f"{'!'*70}")
        print("Q-points with q ≠ 0 require supercell to properly represent")
        print("the phase variation between unit cells. Primitive cell")
        print("displacement may not capture the full physics.")
        print(f"{'!'*70}\n")
    
    # Get original positions in fractional coordinates
    orig_positions = primitive.scaled_positions
    lattice = primitive.cell
    
    for mode_idx in mode_indices:
        freq = frequencies[mode_idx]
        eigvec = eigenvectors[mode_idx]
        
        print(f"\n{'='*60}")
        print(f"Mode {mode_idx+1}: frequency = {freq:.6f} THz")
        print(f"{'='*60}")
        
        # Filter small components
        eigvec_filtered = filter_small_components(eigvec, threshold=1e-6)
        n_nonzero = np.sum(np.abs(eigvec_filtered) > 0)
        print(f"Non-zero components: {n_nonzero}/{len(eigvec_filtered)}")
        
        # Convert to normalized displacements
        displacements_cart = normalize_displacement(eigvec_filtered, masses)
        
        # Check which atoms have significant displacements
        atom_disp_mag = np.linalg.norm(displacements_cart, axis=1)
        active_atoms = np.where(atom_disp_mag > 1e-10)[0]
        print(f"Active atoms (with displacement): {len(active_atoms)}/{n_atoms}")
        print(f"Atom indices: {active_atoms.tolist()}")
        
        # Create structures with different displacement amplitudes
        amplitudes = np.arange(0.01, 0.11, 0.01)  # 0.01 to 0.10 Angstrom
        
        for amp in amplitudes:
            # Scale displacements to desired amplitude (in Cartesian coordinates)
            scaled_disp_cart = displacements_cart * amp
            
            # Convert Cartesian displacements to fractional
            # displacement_frac = displacement_cart @ inv(lattice)
            scaled_disp_frac = scaled_disp_cart @ np.linalg.inv(lattice)
            
            # Apply displacement to original positions
            new_positions = orig_positions + scaled_disp_frac
            
            # Write POSCAR
            # Create filename with q-point identifier
            # Normalize q_name for filename: remove backslash from Greek letters
            if q_name.startswith('\\'):
                # Greek letter like \Gamma, \Lambda, \Delta, etc.
                q_name_clean = q_name[1:]  # Remove backslash
                # Special case: \Gamma -> use simple naming without prefix
                if q_name_clean in ['Gamma', 'GAMMA']:
                    filename = f"POSCAR-mode{mode_idx+1}_amp{amp:.2f}"
                else:
                    filename = f"POSCAR-{q_name_clean}-mode{mode_idx+1}_amp{amp:.2f}"
            elif q_name == 'Gamma':
                filename = f"POSCAR-mode{mode_idx+1}_amp{amp:.2f}"
            elif q_name == 'custom':
                filename = f"POSCAR-custom-mode{mode_idx+1}_amp{amp:.2f}"
            else:
                # Regular label like K, M, L, etc.
                filename = f"POSCAR-{q_name}-mode{mode_idx+1}_amp{amp:.2f}"
            
            write_poscar(primitive, new_positions, filename)
            print(f"  Created {filename} (max displacement: {amp:.3f} Å)")
    
    print(f"\n{'='*60}")
    print("Summary:")
    # Format q_point as clean string
    q_str = f"[{q_point[0]:.6f}, {q_point[1]:.6f}, {q_point[2]:.6f}]"
    print(f"  Q-point: {q_name} {q_str}")
    print(f"  Created {len(mode_indices) * len(amplitudes)} POSCAR files")
    
    # Determine file prefix for display
    if q_name.startswith('\\'):
        q_display = q_name[1:]  # Remove backslash for display
        if q_display in ['Gamma', 'GAMMA']:
            file_prefix = "POSCAR-mode"
            is_gamma = True
        else:
            file_prefix = f"POSCAR-{q_display}-mode"
            is_gamma = False
    elif q_name == 'Gamma':
        file_prefix = "POSCAR-mode"
        is_gamma = True
    elif q_name == 'custom':
        file_prefix = "POSCAR-custom-mode"
        is_gamma = False
    else:
        file_prefix = f"POSCAR-{q_name}-mode"
        is_gamma = False
    
    if is_gamma:
        print(f"  Mode 1: {file_prefix}1_amp0.01 to {file_prefix}1_amp0.10")
        print(f"  Mode 2: {file_prefix}2_amp0.01 to {file_prefix}2_amp0.10")
        print(f"\n  Note: Gamma point modes can be properly represented in primitive cell.")
    else:
        print(f"  Files: {file_prefix}*_amp*")
        print(f"\n  Note: Non-Gamma modes may require supercell for proper representation.")
    
    print(f"\nNext steps:")
    print(f"  1. Relax each structure with VASP")
    print(f"  2. Run: ./analyze_energies.py")
    print(f"  3. Calculate phonons on lowest energy structure")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
