#!/Users/tonyspc/miniforge3/envs/pyxtal/bin/python3
"""
Verify that generated POSCAR files correctly represent Gamma point eigenvector displacements.
"""
import numpy as np
from phonopy import load

def read_poscar_positions(filename):
    """Read atomic positions from POSCAR file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse positions (assuming Direct coordinates, starting at line 8)
    positions = []
    for line in lines[8:]:
        if line.strip():
            vals = line.split()
            if len(vals) >= 3:
                positions.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    return np.array(positions)

def main():
    print("="*70)
    print("VERIFICATION: Checking Gamma point POSCAR files")
    print("="*70)
    
    # Load phonopy data
    phonon = load("phonopy_params.yaml")
    primitive = phonon.primitive
    masses = primitive.masses
    n_atoms = len(masses)
    lattice = primitive.cell
    
    # Read original positions
    orig_positions = primitive.scaled_positions
    
    print(f"\nStructure: {n_atoms} atoms")
    print(f"Chemical symbols: {primitive.symbols}")
    
    # Calculate phonons at Gamma point
    q_point = [0.0, 0.0, 0.0]
    print(f"\n{'='*70}")
    print(f"Calculating phonons at Gamma point: {q_point}")
    print(f"{'='*70}")
    
    phonon.run_qpoints([q_point], with_eigenvectors=True)
    qdict = phonon.get_qpoints_dict()
    
    frequencies = qdict["frequencies"][0]
    eigenvectors = qdict["eigenvectors"][0]
    
    print(f"\nFrequencies (first 6 modes):")
    for i in range(min(6, len(frequencies))):
        print(f"  Mode {i+1}: {frequencies[i]:10.6f} THz")
    
    # Check mode 1
    mode_idx = 0
    eigvec = eigenvectors[mode_idx]
    
    print(f"\n--- Verifying Mode {mode_idx+1} ---")
    print(f"Frequency: {frequencies[mode_idx]:.6f} THz")
    
    # Process eigenvector same way as in disp_origin_struct.py
    eigvec_real = np.real(eigvec.copy())
    eigvec_real[np.abs(eigvec_real) < 1e-6] = 0.0
    
    # Convert to displacements
    eigvec_reshaped = eigvec_real.reshape(n_atoms, 3)
    displacements = np.zeros_like(eigvec_reshaped)
    for i in range(n_atoms):
        displacements[i] = eigvec_reshaped[i] / np.sqrt(masses[i])
    
    # Normalize
    atom_disp_mag = np.linalg.norm(displacements, axis=1)
    max_disp = np.max(atom_disp_mag)
    if max_disp > 0:
        displacements = displacements / max_disp
    
    print(f"Max displacement magnitude before normalization: {max_disp:.6e}")
    print(f"Active atoms: {np.sum(atom_disp_mag > 1e-10)}/{n_atoms}")
    
    print(f"\nChecking generated POSCAR files...")
    
    for amp in [0.01, 0.05, 0.10]:
        filename = f"POSCAR-mode{mode_idx+1}_amp{amp:.2f}"
        
        try:
            # Read displaced positions
            disp_positions = read_poscar_positions(filename)
            
            # Calculate actual displacement
            diff_frac = disp_positions - orig_positions
            diff_cart = diff_frac @ lattice
            actual_disp_mag = np.linalg.norm(diff_cart, axis=1)
            
            # Expected displacement
            expected_disp_cart = displacements * amp
            expected_disp_mag = np.linalg.norm(expected_disp_cart, axis=1)
            
            # Compare
            max_actual = np.max(actual_disp_mag)
            max_expected = amp
            
            print(f"\n  {filename}:")
            print(f"    Max displacement (actual):   {max_actual:.8f} Å")
            print(f"    Max displacement (expected): {max_expected:.8f} Å")
            print(f"    Difference:                  {abs(max_actual - max_expected):.2e} Å")
            
            # Check atom-by-atom
            diff_per_atom = np.abs(actual_disp_mag - expected_disp_mag)
            max_diff = np.max(diff_per_atom)
            
            if max_diff < 1e-6:
                print(f"    Status: ✓ PASS (max difference: {max_diff:.2e} Å)")
            else:
                print(f"    Status: ✗ FAIL (max difference: {max_diff:.2e} Å)")
        
        except FileNotFoundError:
            print(f"\n  {filename}: FILE NOT FOUND")
    
    # Detailed comparison for one case
    print(f"\n{'='*70}")
    print("DETAILED CHECK: POSCAR-mode1_amp0.01 vs Gamma point eigenvector")
    print(f"{'='*70}")
    
    # Read displaced POSCAR
    disp_positions = read_poscar_positions("POSCAR-mode1_amp0.01")
    diff_frac = disp_positions - orig_positions
    diff_cart = diff_frac @ lattice
    actual_disp_mag = np.linalg.norm(diff_cart, axis=1)
    
    expected_disp_cart = displacements * 0.01
    expected_disp_mag = np.linalg.norm(expected_disp_cart, axis=1)
    
    print(f"\nAtom-by-atom comparison (Cartesian displacement magnitudes):")
    print(f"{'Atom':>4} {'Symbol':>6} {'Expected (Å)':>14} {'Actual (Å)':>14} {'Diff (Å)':>14}")
    print("-" * 70)
    
    for i in range(n_atoms):
        symbol = primitive.symbols[i]
        exp = expected_disp_mag[i]
        act = actual_disp_mag[i]
        diff = abs(exp - act)
        marker = " ✓" if diff < 1e-6 else " ✗"
        print(f"{i:4d} {symbol:>6} {exp:14.8f} {act:14.8f} {diff:14.2e}{marker}")
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    max_diff = np.max(np.abs(expected_disp_mag - actual_disp_mag))
    if max_diff < 1e-6:
        print(f"✓ VERIFICATION PASSED: Max difference = {max_diff:.2e} Å")
        print(f"\nGamma point displacements correctly implemented.")
        print(f"Zone-center modes properly represented in primitive cell.")
    else:
        print(f"✗ VERIFICATION FAILED: Max difference = {max_diff:.2e} Å")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
