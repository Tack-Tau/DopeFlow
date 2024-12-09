# DopeFlow: DFT Calculation Workflow for Atomic Substitution Problem

This repository contains scripts for managing VASP calculations on a SLURM-based cluster system.

### 1. job_monitor.sh

A script for submitting and monitoring sequential VASP calculations (Relax → SC → Band → DOS).

#### Usage

```bash
nohup bash job_monitor.sh <calc_type> &
```

where `<calc_type>` can be:
- `Relax`: Structure relaxation
- `SC`: Self-consistent calculation
- `Band`: Band structure calculation
- `DOS`: Density of states calculation

#### Directory Structure
```bash
structure_directory/
├── Relax/
├── SC/
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

## Environment Setup
Ensure these environment variables are set:
- `$AFLOW_HOME`: Path to AFLOW executable
- `$VASPKIT_HOME`: Path to VASPKIT executable
- `$PHONOPY_HOME`: Path to Phonopy executable

## Common Workflow
1. Structure Relaxation (`job_monitor.sh Relax`)
2. Electronic Structure (`job_monitor.sh SC/Band/DOS`)
3. Optical Properties (`redo_optics.sh`)
4. Phonon Calculations (`submit_phonon.sh`)
