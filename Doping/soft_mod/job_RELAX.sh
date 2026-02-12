#!/bin/bash
#SBATCH --job-name=Relax
#SBATCH --partition=Apus,Orion
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=vasp_%j.out
#SBATCH --error=vasp_%j.err

# Load modules
module purge
module load intel/mkl/2024.0 intel/2024 intel-mpi/2021.11
ulimit -s unlimited

# Set environment
export OMP_NUM_THREADS=1
export PMG_VASP_PSP_DIR=$HOME/apps/PBE64

# Intel MPI settings for SLURM
if [ -e /opt/slurm/lib/libpmi.so ]; then
  export I_MPI_PMI_LIBRARY=/opt/slurm/lib/libpmi.so
else
  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0
fi
export I_MPI_FABRICS=shm:ofi

# VASP executable (use srun for SLURM-native MPI launching)
VASP_CMD="srun --mpi=pmi2 $HOME/apps/vasp.6.2.1/bin/vasp_std > log"


# Run VASP
echo "Starting VASP calculation: relax"
echo "Working directory: $(pwd)"
echo "VASP command: $VASP_CMD"
echo "Start time: $(date)"

echo -e "102\n2\n0.04\n"| vaspkit 1> /dev/null
$VASP_CMD

EXIT_CODE=$?

echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# Check if successful
if [ $EXIT_CODE -eq 0 ]; then
    # Verify critical files for Relax calculation
    if [ -f "CONTCAR" ] && [ -s "CONTCAR" ]; then
        echo "VASP calculation completed successfully"
        echo "Verified CONTCAR exists"

        # Clean up large unnecessary files to save disk space
        rm -f CHGCAR CHG WAVECAR WFULL AECCAR* TMPCAR 2>/dev/null

        touch VASP_DONE
    else
        echo "VASP calculation failed: CONTCAR missing/empty"
        # Clean up large intermediate files to save disk space
        rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR 2>/dev/null
        touch VASP_FAILED
    fi
else
    echo "VASP calculation failed with exit code $EXIT_CODE"
    # Clean up large intermediate files to save disk space
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR 2>/dev/null
    touch VASP_FAILED
fi
