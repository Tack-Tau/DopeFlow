#!/Users/tonyspc/miniforge3/envs/pyxtal/bin/python3
"""
Analyze relaxed energies from VASP calculations.
Finds which displacement gives the lowest energy.
"""
import os
import glob
import numpy as np

def extract_energy(outcar_path):
    """Extract final energy from OUTCAR file."""
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        # Find the last "free energy    TOTEN  =" line
        energy = None
        n_ions = None
        
        for line in lines:
            if "free energy    TOTEN  =" in line:
                energy = float(line.split()[4])
            if "NIONS =" in line:
                n_ions = int(line.split()[11])
        
        if energy is None or n_ions is None:
            return None, None
        
        return energy, n_ions
    
    except Exception as e:
        return None, None

def main():
    # Find all directories that might contain relaxed structures
    # Pattern: mode*_amp*/ or gamma_mode*_amp*/ directories
    k_dirs = sorted(glob.glob("mode*_amp*/"))
    gamma_dirs = sorted(glob.glob("gamma_mode*_amp*/"))
    
    dirs = k_dirs + gamma_dirs
    
    if not dirs:
        print("No relaxed calculation directories found.")
        print("Expected directories like: mode1_amp0.01/, gamma_mode1_amp0.01/, etc.")
        return
    
    print(f"Found {len(dirs)} calculation directories\n")
    
    results = []
    
    for d in dirs:
        outcar = os.path.join(d, "OUTCAR")
        
        if not os.path.exists(outcar):
            print(f"  {d:25s} - OUTCAR not found")
            continue
        
        energy, n_ions = extract_energy(outcar)
        
        if energy is None:
            print(f"  {d:25s} - Could not extract energy")
            continue
        
        energy_per_atom = energy / n_ions
        results.append((d, energy, n_ions, energy_per_atom))
        print(f"  {d:25s} E = {energy:12.6f} eV, E/atom = {energy_per_atom:10.6f} eV")
    
    if not results:
        print("\nNo valid results found.")
        return
    
    # Find lowest energy structure
    results.sort(key=lambda x: x[3])  # Sort by energy per atom
    
    print("\n" + "="*70)
    print("Summary (sorted by energy per atom):")
    print("="*70)
    
    for i, (d, e, n, e_per_atom) in enumerate(results):
        marker = " <-- LOWEST ENERGY" if i == 0 else ""
        print(f"{i+1:3d}. {d:25s} E/atom = {e_per_atom:10.6f} eV{marker}")
    
    # Analyze energy differences
    if len(results) > 1:
        e_min = results[0][3]
        e_max = results[-1][3]
        print(f"\nEnergy range: {(e_max - e_min)*1000:.3f} meV/atom")
        
        # Check if all energies are the same (within tolerance)
        if abs(e_max - e_min) < 1e-6:
            print("\nWARNING: All energies are identical!")
            print("This suggests the displacement might not be affecting the energy.")
            print("Possible reasons:")
            print("  1. The soft mode at K point needs supercell expansion")
            print("  2. Try displacing along Gamma point soft modes instead")
            print("  3. Check if the structure is actually relaxing")
    
    print("\n" + "="*70)
    print(f"Lowest energy structure: {results[0][0]}")
    print("="*70)
    print("\nNext step:")
    print(f"  cd {results[0][0].rstrip('/')}")
    print(f"  # Calculate Gamma point phonons on CONTCAR")

if __name__ == "__main__":
    main()
