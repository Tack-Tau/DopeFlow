# Soft Mode Displacement Structure Generator

A tool to resolve negative phonon frequencies (soft modes) by generating displaced structures along eigenvector directions for relaxation and energy analysis.

## Overview

This workflow helps identify the true ground state structure when phonon calculations show imaginary (negative) frequencies. Instead of using phonopy's MODULATION with large supercells, this approach:

1. Extracts eigenvectors at user-selected q-points
2. Displaces atoms in the primitive cell along eigenvector directions
3. Generates structures with varying displacement amplitudes (0.01-0.10 Å)
4. Allows energy comparison to find the lowest energy configuration

## Quick Start

```bash
# Phase 1: Generate and relax displaced structures
./disp_origin_struct.py              # Generate POSCAR-mode*_amp* files
# ... set up and run VASP relaxations ...

# Phase 2: Analyze and set up phonon calculation
./analyze_phon.py                    # Find lowest energy structure
./setup_phonon_calc.sh               # Automated phonon calculation setup

# Phase 3: Run phonon calculations
cd PHON_relaxed
./submit_all.sh                      # Submit displacement calculations
# ... wait for jobs to complete ...

# Phase 4: Post-process results
./postprocess.sh                     # Collect forces and generate plots
```

**See `WORKFLOW_GUIDE.md` for detailed step-by-step instructions.**

## Files Description

### Main Scripts

- **`disp_origin_struct.py`** - Interactive structure generator with q-point selection
- **`analyze_phon.py`** - Comprehensive analysis: energies, phonon post-processing, visualization
- **`setup_phonon_calc.sh`** - Automated phonon calculation setup script
- **`analyze_energies.py`** - Simple energy analyzer (basic version)
- **`verify_displacements.py`** - Verification script for generated displacements
- **`check_freq.py`** - Quick frequency checker

### Documentation

- **`README.md`** - This file (overview and quick reference)
- **`WORKFLOW_GUIDE.md`** - Detailed step-by-step workflow guide
- **`FINAL_SUMMARY.txt`** - Quick reference summary

### Generated Files

- **`POSCAR-mode*_amp*`** - Displaced structures (20 files by default)
  - 10 files for mode 1 (amplitudes 0.01 to 0.10 Å)
  - 10 files for mode 2 (amplitudes 0.01 to 0.10 Å)

## Interactive Q-point Selection

The script now provides **interactive q-point selection** with automatic detection of high-symmetry points using pymatgen:

```
Q-POINT SELECTION
======================================================================

High-symmetry q-points available:
     Label       Fractional Coordinates
----------------------------------------------------------------------
  1.  Gamma     0.000000    0.000000    0.000000
  2.      A     0.000000    0.000000    0.500000
  3.      H     0.333333    0.333333    0.500000
  4.      K     0.333333    0.333333    0.000000
  5.      L     0.500000    0.000000    0.500000
  6.      M     0.500000    0.000000    0.000000

Options:
  - Enter number (1-6) to select from list
  - Enter 'custom' to input custom coordinates
  - Press Enter for Gamma point [0, 0, 0]

Your choice: 
```

**Note on Greek Letters:** Labels like `\Gamma`, `\Lambda`, `\Delta`, etc. are automatically handled - the backslash is removed from filenames (e.g., `\Lambda` → `POSCAR-Lambda-mode*`). Gamma point files have no prefix (e.g., `POSCAR-mode*`).

### Recommendations for Q-point Selection

#### Gamma Point (q = [0, 0, 0]) - Recommended
- **Best for**: Primitive cell displacements
- **Physics**: Zone-center modes, all unit cells displace identically
- **Advantages**:
  - Can be properly represented in primitive cell
  - No phase variation between cells
  - Consistent with [2 2 2] supercell phonon calculations
- **Use when**: Your supercell calculation shows Gamma point soft modes

#### Non-Gamma Points (q ≠ 0) - Use with Caution
- **Physics**: Requires phase variation between unit cells
- **Limitations**:
  - Primitive cell displacement is physically incomplete
  - True representation requires supercell (e.g., 3×3×1 for K point)
  - May not show energy variation after relaxation
- **Use when**: Testing or comparison purposes only

### Example: K Point Issue

For K point (q = [1/3, 1/3, 0]):
- Phonopy MODULATION with 3×3×1 supercell gave identical energies
- Indicates K point mode doesn't drive structural distortion
- Primitive cell displacement misses phase variation

## Problem Context

### Your Ca10P6 Structure

Soft modes observed:
- **Gamma point**: -1.018 THz (modes 1 and 2)
- **K point**: -1.389 THz (mode 1)

Previous attempts:
- Phonopy MODULATION on K point (3×3×1 supercell) gave identical energies
- Suggests K point mode doesn't lower energy

## Technical Details

### Eigenvector Processing Algorithm

1. **Load phonopy data**: Read from `phonopy_params.yaml`
2. **Calculate at q-point**: Get eigenvectors at user-selected q-point
3. **Extract eigenvector**: Complex array from phonopy
4. **Take real part**: `eigvec_real = np.real(eigvec)`
5. **Filter noise**: Set components < 1e-6 to zero
6. **Remove mass weighting**: `disp[i] = eigvec[i] / sqrt(mass[i])`
7. **Normalize**: Scale so maximum displacement = 1.0 Å
8. **Scale to amplitude**: Multiply by target amplitude (0.01-0.10 Å)
9. **Apply to structure**: Add displacements in fractional coordinates

### Displacement Verification

All generated structures are verified for correctness:
```
POSCAR-mode1_amp0.01: max displacement = 0.010 Å ✓
POSCAR-mode1_amp0.05: max displacement = 0.050 Å ✓
POSCAR-mode1_amp0.10: max displacement = 0.100 Å ✓

Maximum error: 5.66×10⁻¹⁶ Å (numerical precision)
```

Run verification:
```bash
./verify_displacements.py
```

### Active Atoms (Gamma Point, Mode 1)

For the Ca10P6 structure:
- **12 out of 16 atoms** have significant displacements
- Ca atoms: 0, 3, 5, 6, 7, 9 (indices from 0)
- P atoms: 10, 11, 12, 13, 14, 15

## Complete Workflow

### Phase 1: Generate and Relax Displaced Structures

1. **Generate displaced structures:**
   ```bash
   ./disp_origin_struct.py
   # Press Enter to select Gamma point (recommended)
   ```

2. **Set up VASP relaxations:**
   ```bash
   for poscar in POSCAR-mode1_amp*; do
       name=${poscar#POSCAR-}
       mkdir -p $name
       cp $poscar $name/POSCAR
       cp INCAR_RELAX KPOINTS POTCAR $name/
       # Submit job...
   done
   ```

3. **Analyze energies:**
   ```bash
   ./analyze_phon.py
   # This identifies the lowest energy structure
   ```

### Phase 2: Phonon Calculation on Optimal Structure

4. **Set up phonon calculation on lowest energy structure:**
   ```bash
   # Example: if mode1_amp0.07 is lowest
   mkdir PHON_relaxed
   cd PHON_relaxed
   
   # Copy relaxed structure
   cp ../mode1_amp0.07/CONTCAR POSCAR
   
   # Generate phonopy displacements (same supercell as original)
   phonopy -d --dim="2 2 2"
   # This creates SPOSCAR and POSCAR-{001..N}
   
   # Create Gamma-only KPOINTS for force calculations
   cat > KPOINTS << EOF
   Gamma-point only
   0
   Monkhorst-Pack
   1 1 1
   0 0 0
   EOF
   
   # Run VASP for each displacement
   for disp in POSCAR-{001..010}; do
       dir=${disp#POSCAR-}
       mkdir $dir
       cp $disp $dir/POSCAR
       cp KPOINTS INCAR_PHON POTCAR $dir/
       # Submit VASP job...
   done
   ```

5. **Collect forces and create force constants:**
   ```bash
   cd PHON_relaxed
   
   # After all jobs complete
   phonopy -f */vasprun.xml
   # This creates FORCE_SETS
   
   # Generate force constants
   phonopy --dim="2 2 2" -c POSCAR
   # This creates phonopy_params.yaml
   ```

6. **Post-process phonon results:**
   ```bash
   cd PHON_relaxed
   ../analyze_phon.py
   ```

This generates:
- **Force constants:**
  - `FORCE_SETS` - Collected forces
  - `phonopy_params.yaml` - Force constants from phonopy
- **Data files (ElectrideFlow compatible):**
  - `phonon_band.dat` - Band structure data
  - `band_kpath.dat` - K-path metadata
  - `phonon_dos.dat` - DOS data
- **Plots:**
  - `phonon_band_dos.png` - Combined plot
  - `phonon_band.png` - Band structure only
  - `phonon_dos.png` - DOS only
- **Console output:**
  - Gamma point frequency check

### Expected Results

#### If Energies Vary (Your Case!)
- One displacement amplitude gives lower energy
- Structure wants to distort in that direction
- Use lowest energy structure for phonon calculation
- Soft mode should be reduced or eliminated

#### If All Energies Are Identical
Possible reasons:
1. Soft mode doesn't drive structural distortion
2. Need larger displacement amplitudes (try 0.15-0.20 Å)
3. Need to combine multiple soft modes
4. Force constants need tighter convergence
5. Structure at saddle point requiring specific symmetry breaking

## Analysis Scripts

### Energy Analysis (`analyze_phon.py`)

After relaxations complete:

```bash
./analyze_phon.py
```

Output example:
```
Found 20 relaxed structures

======================================================================
Energy Analysis (sorted by energy per atom)
======================================================================
  1. mode1_amp0.07         E/atom = -4.305964 eV <-- LOWEST ENERGY
  2. mode1_amp0.08         E/atom = -4.305964 eV (+0.00 meV)
  3. mode1_amp0.09         E/atom = -4.305964 eV (+0.00 meV)
  ...

Energy range: 3.758 meV/atom
Lowest energy structure: mode1_amp0.07
======================================================================

NEXT STEPS:
1. Use the relaxed structure from: mode1_amp0.07/CONTCAR
2. Set up phonon calculation...
```

### Phonon Post-Processing (`analyze_phon.py --collect-forces`)

After displacement calculations complete, run directly in the phonon calculation directory:

```bash
cd PHON_relaxed  # or your phonon calculation directory with 001/, 002/, etc.
../analyze_phon.py --collect-forces
```

This will:
- Collect forces from `*/vasprun.xml`
- Generate `FORCE_SETS` and `phonopy_params.yaml`
- Check Gamma point frequencies
- Generate data files: `phonon_band.dat`, `band_kpath.dat`, `phonon_dos.dat`
- Generate plots: `phonon_band.png`, `phonon_dos.png`, `phonon_band_dos.png`
- Report if soft modes are eliminated

**Auto-detection:** If you run `analyze_phon.py` in a directory containing `phonopy_disp.yaml` and numbered directories (`001/`, `002/`, etc.), it automatically enters phonon processing mode.

Output:
```
======================================================================
PHONON CALCULATION POST-PROCESSING
======================================================================
Working directory: /path/to/PHON_relaxed

Collecting forces from displacement calculations...
  Found 96 displacement directories
  Found 96 vasprun.xml files

  Running: phonopy -f */vasprun.xml
  Created: FORCE_SETS

  Running: phonopy -c POSCAR
  Created: phonopy_params.yaml

Loading phonopy data...
  Structure: Ca5P3 (16 atoms)

Checking Gamma point frequencies...
  Gamma point frequencies (first 10 modes):
    Mode  1:   0.123456 THz
    Mode  2:   0.234567 THz
    ...
  
  SUCCESS: All soft modes eliminated!

Generating data files...
    Saved: phonon_band.dat
    Saved: band_kpath.dat

Calculating DOS (mesh: [20, 20, 20])...
    Saved: phonon_dos.dat

Generating band structure plot...
    Saved: phonon_band.png
  
Generating DOS plot...
    Saved: phonon_dos.png
  
Generating combined band+DOS plot...
    Saved: phonon_band_dos.png

======================================================================
PHONON POST-PROCESSING COMPLETE
======================================================================
All soft modes eliminated!

Generated files:
  - phonopy_params.yaml     (force constants)
  - phonon_band.dat         (band structure data)
  - band_kpath.dat          (k-path metadata)
  - phonon_dos.dat          (DOS data)
  - phonon_band.png         (band plot)
  - phonon_dos.png          (DOS plot)
  - phonon_band_dos.png     (combined plot)
======================================================================
```

**Data files compatible with [ElectrideFlow plotting](https://github.com/MaterSim/ElectrideFlow):**
- `phonon_band.dat` - Band structure data
- `band_kpath.dat` - K-path metadata (lattice, segments, labels)
- `phonon_dos.dat` - DOS data with element projections

These can be used with `phonon_band_dos_plot.py` for local plotting without rerunning calculations.

## Physics Background

### Phonon Mode Formula

Atomic displacement in a phonon mode:
```
u_j(l) = (A/√m_j) × Re[e_j × exp(i·q·R_l)]
```

Where:
- `A` = amplitude
- `m_j` = mass of atom j
- `e_j` = eigenvector component
- `q` = wave vector
- `R_l` = position of unit cell l
- `exp(i·q·R_l)` = phase factor

### Why Q-point Matters

**Gamma point (q = 0):**
```
exp(i·q·R_l) = exp(0) = 1  (for all cells)
→ All cells displace identically
→ Can be represented in primitive cell ✓
```

**K point (q = [1/3, 1/3, 0]):**
```
exp(i·q·R_l) varies periodically
→ Different cells have different phases
→ Requires 3×3×1 supercell for proper representation
→ Primitive cell displacement is incomplete ✗
```

## Troubleshooting

### Pymatgen Not Installed

If you see this error:
```
ERROR: Pymatgen is required but not installed
```

Install pymatgen:
```bash
# Using conda (recommended)
conda install -c conda-forge pymatgen

# Using pip
pip install pymatgen
```

### Large Displacements Needed

If standard amplitudes (0.01-0.10 Å) don't show energy variation, modify the script:

```python
# In disp_origin_struct.py, line ~155:
amplitudes = np.arange(0.01, 0.11, 0.01)  # Default

# Change to:
amplitudes = np.arange(0.05, 0.21, 0.01)  # Larger amplitudes
```

### Combining Multiple Modes

To displace along multiple soft modes simultaneously:
1. Generate structures for mode 1
2. Generate structures for mode 2
3. Manually create combined structures by adding displacements

## References

- **Phonopy Documentation**: https://phonopy.github.io/phonopy/
- **Phonopy MODULATION**: https://phonopy.github.io/phonopy/setting-tags.html#modulation
- **Soft Modes & Phase Transitions**: Soft modes often indicate structural phase transitions where the true ground state has lower symmetry

## Citation

If you use this workflow, please cite:
- **Phonopy**: A. Togo and I. Tanaka, Scr. Mater. **108**, 1-5 (2015)
- **Pymatgen** (if used): S.P. Ong et al., Comput. Mater. Sci. **68**, 314-319 (2013)

## Verification Status

**All displacement implementations verified correct**
- Numerical precision: < 1×10⁻¹⁵ Å
- Tested at multiple amplitudes (0.01, 0.05, 0.10 Å)
- All 16 atoms checked individually
- Mass-weighted eigenvector conversion confirmed

## Version History

- **v2.0** - Added interactive q-point selection with pymatgen integration
- **v1.5** - Switched default to Gamma point based on [2 2 2] supercell
- **v1.0** - Initial K point implementation

## License & Support

This is a research tool provided as-is. For questions or issues:
1. Check the verification script output
2. Review the physics background section
3. Consult phonopy documentation

---

**Last Updated**: January 2026
