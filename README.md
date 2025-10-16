# DopeFlow: DFT Calculation Workflow for Atomic Substitution Problem

This repository contains scripts for managing VASP calculations on a SLURM-based cluster system.

### 1. job_monitor.sh

A script for submitting and monitoring sequential VASP calculations (Relax → SC → ELF → Band → DOS).

#### Usage

```bash
nohup bash job_monitor.sh <calc_type> &
```

where `<calc_type>` can be:
- `Relax`: Structure relaxation
- `SC`: Self-consistent calculation
- `ELF`: Electron Localization Function calculation
- `Band`: Band structure calculation
- `DOS`: Density of states calculation

#### Directory Structure
```bash
structure_directory/
├── Relax/
├── SC/
├── ELF/
├── DOS/
└── Band/
```

#### Required Files
- `INCAR_<calc_type>`: INCAR file for each calculation type
- `sbp_<calc_type>.sh`: SLURM submission script for each calculation type
- `POTCAR`: VASP pseudopotential file
- `../aflow_sym/uniq_poscar_list`: List of structures to process
- `diverge_structs`: (optional) List of structures to skip

### 2. redo_optics.sh/restart_optics.sh

Scripts for managing optical calculations (SC → DIAG → GW0 → BSE) with automatic error checking and resubmission.

#### Usage

For normal execution with error checking:
```bash
nohup ./redo_optics.sh > redo_optics.log 2>&1 &
```

For forced restart of all calculations:
```bash
nohup ./restart_optics.sh > restart_optics.log 2>&1 &
```

#### Required Files
- `Direct_dir`: File containing list of directories to process
- INCAR files:
  - `INCAR_SC`
  - `INCAR_DIAG`
  - `INCAR_GW0`
  - `INCAR_BSE`
- SLURM submission scripts:
  - `sbp_SC.sh`
  - `sbp_DIAG.sh`
  - `sbp_GW0.sh`
  - `sbp_BSE.sh`
- `POTCAR_GW`: VASP GW pseudopotential file

#### `<calc_type>` can be:
- `SC`: DFT groundstate calculation
- `DIAG`: DFT "virtual" orbitals (empty states)
- `GW0`: RPA quasiparticles with single-shot GW
- `BSE`: BSE calculation

#### Directory Structure

```bash
structure_directory/
├── Optics/
│ ├── SC/
│ ├── DIAG/
│ ├── GW0/
│ └── BSE/
```

#### Output Logs
For redo_optics.sh:
- `optical_jobs.log`: Detailed job submission information
- `job_<calc_type>.log`: Job counting logs for each calculation type

For restart_optics.sh:
- `restart_optical_jobs.log`: Detailed job submission information for restarts
- `job_<calc_type>_restart.log`: Job counting logs for restarted calculations

#### Features

redo_optics.sh:
- Automatic error detection and job resubmission
- Sequential dependency handling
- Detailed logging of job submissions
- Limits concurrent jobs up to 60 computational nodes

restart_optics.sh:
- Forces restart of all calculations regardless of previous status
- Maintains same workflow and dependencies
- Uses separate log files to avoid confusion with original runs
- Limits concurrent jobs up to 60 computational nodes

## Features

### job_monitor.sh
- Manages sequential job submissions
- Limits concurrent jobs to 60
- Handles failed calculations
- Supports structure skipping via `diverge_structs`

### redo_optics.sh/restart_optics.sh
- Automatic directory creation and management
- Sequential dependency handling (SC → DIAG → GW0 → BSE)
- Automatic error detection and job resubmission
- Detailed logging of job submissions
- Limits concurrent jobs to 60

## Common Issues
1. Missing required files - ensure all INCAR and submission scripts are present
2. Directory permissions - ensure write access in all directories
3. SLURM queue limits - script will wait if queue is full
4. Failed calculations - check individual VASP output files for errors
5. Missing vasprun.xml - script will detect and resubmit affected calculations
6. Failed phonon calculations - use get_err_phon.sh to generate resubmission script

## Notes
- Both scripts assume SLURM job scheduler
- Maximum concurrent jobs is set to 60
- Scripts will create necessary directories if they don't exist
- Error handling includes automatic resubmission of failed jobs

### 3. submit_phonon.sh

A script for managing phonon calculations with automatic supercell generation and job monitoring.

#### Usage

```bash
chmod +x submit_phonon.sh
nohup ./submit_phonon.sh > nohup.out 2>&1 &
```

#### Required Files
- `phonon_list`: File containing list of directories to process
- `INCAR_PHON`: INCAR file for phonon calculations
- `sbp_PHON.sh`: SLURM submission script for phonon calculations
- Supporting scripts:
  - `convert_kpath.sh`
  - `generate_supercell.sh`
  - `extract_band_conf.sh`
  - `preprocess_high_symmetry_points.sh`

#### Directory Structure

```bash
structure_directory/
├── Relax/
│ └── CONTCAR
└── PHON/
├── POSCAR-*
├── INCAR
├── POTCAR
└── sbp.sh
```

#### Output Logs
- `job_PHON.log`: Detailed job submission tracking
- Records which phonon calculations have been submitted for each structure

#### Features
- Automatic supercell generation using VASPKIT
- Batch submission (10 jobs at a time)
- Limits concurrent jobs to 50
- Resumes from last submitted job if interrupted
- Maintains submission history in log file

#### Workflow
1. Reads structures from `phonon_list`
2. For each structure:
   - Creates PHON directory
   - Copies CONTCAR from Relax directory
   - Generates primitive cell using VASPKIT
   - Generates supercells
   - Submits jobs in batches
3. Monitors job queue and maintains submission limits
4. Tracks progress in log file

#### Tips for Phonon Calculations
1. Check supercell size in `generate_supercell.sh`
2. Monitor convergence in individual phonon calculations
3. Use `job_PHON.log` to track submission progress
4. Check VASPKIT output for primitive cell generation

### 4. Post-Processing Scripts

#### post-proc_phonon.sh
A script for post-processing phonon calculations with automatic error detection and data generation.

#### Usage
```bash
sbatch phonon-pp-job.sh
```

#### Features
- Automatic error detection in SLURM output files
- Generates FORCE_SETS using phonopy
- Creates phonon band plots and raw data files
- Handles LaTeX formatting for band labels
- Detailed logging with configurable verbosity

#### Required Files
- `phonon_list`: List of structures to process
- Supporting scripts:
  - `convert_kpath.sh`
  - `extract_band_conf.sh`
  - `preprocess_high_symmetry_points.sh`

#### band_gap-pp.sh
A script for analyzing and categorizing band structures based on their electronic properties.

#### Usage
```bash
./band_gap-pp.sh
```

#### Features
- Automatically categorizes structures as Direct, Indirect, or Metallic
- Uses VASPKIT for band structure analysis
- Error detection in SLURM output files
- Generates categorized lists of structures

#### Output Files
- `Direct_dir`: List of structures with direct band gaps
- `Indirect_dir`: List of structures with indirect band gaps
- `Metallic_dir`: List of structures with metallic/semimetallic band gaps

#### get_err_phon.sh
A utility script for handling failed phonon calculations.

#### Usage
```bash
./get_err_phon.sh
```

#### Features
- Analyzes phonon post-processing logs for errors
- Generates resubmission script for failed calculations
- Handles missing or corrupted vasprun.xml files
- Automatic cleanup and job resubmission

### 5. ELF Analysis for Electride Detection

#### analyze_electride.py
A Python script for analyzing Electron Localization Function (ELF) calculations to identify potential electride structures using **Bader topological analysis**. Electrides are materials where electrons occupy interstitial regions rather than being associated with atoms.

Uses Bader topological analysis from the [Henkelman group](https://theory.cm.utexas.edu/henkelman/code/bader/) to identify critical points in the ELF field, avoiding false positives from covalent bond regions.

#### Usage

**Single structure analysis:**
```bash
cd /path/to/structure/ELFCAR
python3 /path/to/aflow_sym/analyze_electride.py ELFCAR
```

**Batch analysis of all structures:**
```bash
cd /path/to/parent/directory
python3 /path/to/aflow_sym/analyze_electride.py --batch . -o electride_results.csv
```

**With custom parameters:**
```bash
python3 analyze_electride.py ELFCAR --threshold 0.7 --min-distance 2.0 --volume-threshold 0.5
```

**With custom bader executable:**
```bash
python3 analyze_electride.py --bader-exe /path/to/bader /path/to/structure/ELFCAR
```

**Force regenerate BCF.dat:**
```bash
rm /path/to/ELF/BCF.dat
python3 analyze_electride.py /path/to/structure/ELFCAR
```

#### Required Files
- `ELFCAR`: Output from VASP ELF calculation (generated with `LELF=.TRUE.` in INCAR)
- `bader` executable: Download from [Henkelman Group](https://theory.cm.utexas.edu/henkelman/code/bader/)
  - **Auto-detection**: Script automatically looks for `bader` in the same directory as ELFCAR
  - Alternative: Add `bader` to system PATH or use `--bader-exe` option

#### Features
- **Bader topological analysis**: Rigorous critical point detection in ELF field
- **Automatic electride detection** based on interstitial ELF maxima
- **Zero false positives**: Correctly distinguishes covalent bonds from interstitial electrons
- **Distance-based filtering** to exclude atomic regions
- **Volume estimation** of electron-rich interstitial regions
- **Batch processing** for analyzing multiple structures
- **CSV export** for systematic analysis
- **BCF.dat caching**: Reuses existing Bader analysis results

#### Output
The script provides:
- Potential electride classification (yes/no)
- Maximum ELF value in interstitial regions
- Number of interstitial electron sites
- Volume and volume fraction of interstitial regions
- Distance of interstitial sites from nearest atoms

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.6 | Minimum ELF value for electride detection |
| `--min-distance` | 1.5 Å | Minimum distance from atoms to consider interstitial |
| `--volume-threshold` | 0.5 Å³ | Minimum volume for significant interstitial region |
| `--output` | electride_analysis.csv | Output file for batch analysis |


#### Tips for ELF Calculations
1. **INCAR settings**: Ensure `LELF=.TRUE.` is set in `INCAR_ELF`
2. **Grid density**: Use fine FFT grids for accurate ELF calculations
3. **Convergence**: ELF calculations should be performed on well-converged charge densities
4. **ELFCAR vs CHGCAR**: 
   - We analyze `ELFCAR` (ELF field) directly with `bader ELFCAR`
   - The `-ref CHGCAR_sum` option is for charge density analysis, not needed for ELF
   - VASP outputs complete ELF field in ELFCAR, no core correction needed
5. **BCF.dat caching**: 
   - The script reuses existing BCF.dat if present (faster)
   - Delete BCF.dat to force regeneration after changes
6. **Bader executable detection** (in order of priority):
   - User-specified path via `--bader-exe`
   - `bader` file in same directory as ELFCAR (convenient for per-structure executables)
   - `bader` in system PATH
7. **Threshold tuning**: Adjust `--threshold` based on your material system:
   - Strong electrides: ELF > 0.7
   - Moderate electrides: ELF 0.5-0.7
   - Weak localization: ELF < 0.5
8. **Distance parameter**: `--min-distance 1.5` (default) works for most cases
   - Increase to 2.0 Å for systems with large atoms
   - Decrease to 1.2 Å for compact structures

#### Workflow for ELF Analysis
```bash
# 1. Submit ELF calculations
nohup bash job_monitor.sh ELF &

# 2. Wait for calculations to complete

# 3. Analyze single structure
python3 ../aflow_sym/analyze_electride.py 1/ELF/ELFCAR

# 4. Or batch analyze all structures
python3 ../aflow_sym/analyze_electride.py --batch . -o electride_results.csv

# 5. Check results
cat electride_results.csv
grep "True" electride_results.csv  # List potential electrides
```
## Script Dependencies
- [AFLOW](https://aflowlib.org/)
- [VASPKIT](https://vaspkit.com/index.html)
- [Phonopy](https://phonopy.github.io/phonopy/)

### Python Dependencies
Depending on which doping script you use, you'll need different Python packages:

For general substitution **WITHOUT** symmetry bias: \
**Example:** `aflow_sym/rnd_SiGe_doping.py` or `aflow_sym/NaSiGe_doping.py`
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html)

For using Fingerprint energy as symmetry bias: \
**Example:** `aflow_sym/Doping.py`
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
- [libfp](https://github.com/Rutgers-ZRG/libfp)

For explicitly using group-subgroup splitting: \
**Example:** `aflow_sym/subgroup_doping.py`
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
- [Pymatgen](https://pymatgen.org/)
- [PyXtal](https://pyxtal.readthedocs.io/en/latest/index.html)

**For FP-diff PCA clustering with duplicate avoidance (RECOMMENDED):** \
**Example:** `aflow_sym/fp_cluster_doping.py`
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
- [libfp](https://github.com/Rutgers-ZRG/libfp)
- [scikit-learn](https://scikit-learn.org/) (for HDBSCAN clustering)
- [matplotlib](https://matplotlib.org/) (for visualization)
- [scipy](https://scipy.org/)
- [numba](https://numba.pydata.org/)
- [kimpy](https://github.com/openkim/kimpy) (optional, for KIM energy filtering)

**For ELF electride analysis:** \
**Example:** `aflow_sym/analyze_electride.py`
- [Pymatgen](https://pymatgen.org/) (for reading ELFCAR files)
- [scipy](https://scipy.org/) (for local maxima detection and spatial analysis)
- [numpy](https://numpy.org/)

## Environment Setup
Ensure these environment variables are set:
- `$AFLOW_HOME`: Path to AFLOW executable
- `$VASPKIT_HOME`: Path to VASPKIT executable
- `$PHONOPY_HOME`: Path to Phonopy executable

## Common Workflow
1. Structure Generation (`fp_cluster_doping.py`)
2. Structure Relaxation (`job_monitor.sh Relax`)
3. Electronic Structure (`job_monitor.sh SC/Band/DOS`)
4. ELF Analysis (Optional, `job_monitor.sh ELF` + `analyze_electride.py`)
5. Optical Properties (`redo_optics.sh`)
6. Phonon Calculations (`submit_phonon.sh`)

## 5. Atomic Substitution with FP-Diff PCA Clustering

### fp_cluster_doping.py (RECOMMENDED)

A sophisticated method for generating diverse atomic substituted structures with **automatic duplicate avoidance** using PCA on fingerprint difference matrices and HDBSCAN clustering.

#### Key Features

- **84% uniqueness validated by AFLOW** (100% for 3+ substitutions)
- **PCA on fingerprint differences** preserves full structural information
- **HDBSCAN clustering** in principal component space
- **Optional KIM energy filtering** removes unstable structures
- **Automatic visualization** in PCA space
- **Adaptive parameters** for different dataset sizes

#### Usage

**Command Line:**
```bash
cd aflow_sym/
python3 fp_cluster_doping.py
```

You will be prompted for:
- Element to substitute (e.g., `Si`)
- New element (e.g., `Ge`)
- Maximum number of atoms to substitute
- Maximum structures per substitution level
- Number of PCA components (default: 10)
- Whether to use KIM energy filtering (y/n)
- Whether to generate visualization plots (y/n)

**Python API:**
```python
import ase.io
from fp_cluster_doping import POSCAR_GEN_CLUSTER

# Load structure
atoms = ase.io.read('POSCAR')

# Generate diverse structures
structures = POSCAR_GEN_CLUSTER(
    atoms_origin=atoms,
    elem_from='Si',
    elem_to='Ge',
    max_subs=5,
    max_structures=10,
    visualize=True,
    n_pca_components=10,
    kim_model="Tersoff_LAMMPS_Tersoff_1989_SiGe__MO_350526375143_004"
)
```

#### Algorithm Overview

1. **Generate Candidates**: Create random substitutions for each level
2. **Compute FP Differences**: Calculate aligned fingerprint differences using Hungarian algorithm
3. **PCA Transformation**: Project to principal component space
4. **HDBSCAN Clustering**: Identify structural clusters using Euclidean distance
5. **Smart Selection**: Choose centroids + half-radius structures + outliers
6. **Energy Filtering** (optional): Exclude high-energy structures using KIM calculator

#### Output Files

**POSCAR Files:**
- `POSCAR_N_M` where N = substitution level, M = structure index
- Example: `POSCAR_3_5` = 5th structure with 3 substitutions

**Visualization:**
- `selected_N_substitutions.png` - 2D projection in first 2 PC dimensions
- Shows cluster centroids (stars), half-radius points (circles), and outliers (diamonds)
- Note: Always plots first 2 PCs regardless of `n_pca_components` used for clustering

#### Performance

Validated with AFLOW `--compare_materials`:

| Substitutions | Uniqueness | Status |
|---------------|------------|--------|
| 1 atom | 30% |   Expected* |
| 2 atoms | 90% |   Excellent |
| 3+ atoms | 100% |   Perfect |

\* *Low uniqueness for single substitutions is expected: high-symmetry structures (e.g., diamond Si) have only ~3 symmetrically distinct sites. The algorithm correctly identifies all distinct sites.*

#### Parameters Guide

**n_pca_components**: Number of principal components for clustering
- Default: 10 (captures ~78% variance for Si-Ge)
- Higher values capture more variance but may overfit
- Suggested range: 5-30
  - 5 components: ~69% variance, faster
  - 10 components: ~78% variance, balanced (recommended)
  - 20 components: ~90% variance, more detailed
  - 30 components: ~98% variance, very detailed

**KIM model** (optional energy filtering):
- Si-Ge systems: `"Tersoff_LAMMPS_Tersoff_1989_SiGe__MO_350526375143_004"`
- Excludes top 20% highest energy structures
- Requires [kimpy](https://github.com/openkim/kimpy) installation
- Use `None` to disable

#### Tips

1. **For high-symmetry structures**: Expect low uniqueness for single substitutions (this is correct!)
2. **For complex substitutions**: Algorithm achieves near-perfect duplicate avoidance
3. **Dataset size**: Generate 3× `max_structures` candidates for best diversity
4. **Visualization**: Use to verify cluster quality and selection strategy

#### Duplicate Checking

To verify uniqueness of generated structures:
```bash
bash reduce_sim_struct.sh
cat uniq_poscar_list
```

This uses AFLOW to identify symmetrically equivalent structures.
