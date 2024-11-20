# VASP Job Submission Scripts

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

#### Required Files
- `INCAR_<calc_type>`: INCAR file for each calculation type
- `sbp_<calc_type>.sh`: SLURM submission script for each calculation type
- `POTCAR`: VASP pseudopotential file
- `../aflow_sym/uniq_poscar_list`: List of structures to process
- `diverge_structs`: (optional) List of structures to skip

### 2. redo_optics.sh

A script for managing optical calculations (SC → DIAG → GW0 → BSE) with automatic error checking and resubmission.

#### Usage

```bash
chmod +x redo_optics.sh
nohup ./redo_optics.sh > nohup.out 2>&1 &
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

#### Directory Structure

```bash
structure_directory/
├── Optics/
│ ├── DIAG/
│ ├── GW0/
│ └── BSE/
├── Relax/
└── SC/
```

#### Output Logs
- `optical_jobs.log`: Detailed job submission information
- `job_<calc_type>.log`: Job counting logs for each calculation type
- `nohup.out`: General script output

## Features

### job_monitor.sh
- Manages sequential job submissions
- Limits concurrent jobs to 60
- Handles failed calculations
- Supports structure skipping via `diverge_structs`

### redo_optics.sh
- Automatic directory creation and management
- Sequential dependency handling (SC → DIAG → GW0 → BSE)
- Automatic error detection and job resubmission
- Detailed logging of job submissions
- Limits concurrent jobs to 60

## Tips
1. Always test with a small set of structures first
2. Monitor the log files for progress and errors
3. Use `squeue -u $USER` to check running jobs
4. Check `nohup.out` for script execution details

## Common Issues
1. Missing required files - ensure all INCAR and submission scripts are present
2. Directory permissions - ensure write access in all directories
3. SLURM queue limits - script will wait if queue is full
4. Failed calculations - check individual VASP output files for errors

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
├── POSCAR- # Generated supercell POSCARs
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

[Rest of the README remains the same...]

## Script Dependencies
- VASPKIT
- Phonopy

## Environment Setup
Ensure these environment variables are set:
- `$VASPKIT_HOME`: Path to VASPKIT executable
- `$PHONOPY_HOME`: Path to Phonopy executable

## Common Workflow
1. Structure Relaxation (`job_monitor.sh Relax`)
2. Electronic Structure (`job_monitor.sh SC/Band/DOS`)
3. Optical Properties (`redo_optics.sh`)
4. Phonon Calculations (`submit_phonon.sh`)
