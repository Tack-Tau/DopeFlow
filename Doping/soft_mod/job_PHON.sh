#!/bin/bash
#SBATCH --job-name=PHON
#SBATCH --partition=Apus,Orion
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
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

# Intel MPI settings
if [ -e /opt/slurm/lib/libpmi.so ]; then
  export I_MPI_PMI_LIBRARY=/opt/slurm/lib/libpmi.so
else
  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0
fi
export I_MPI_FABRICS=shm:ofi

# VASP executable
VASP_CMD="srun --mpi=pmi2 $HOME/apps/vasp.6.2.1/bin/vasp_std"

echo "Start time: $(date)"
echo ""


if [ ! -f "POSCAR" ] || [ ! -s "POSCAR" ]; then
    echo "ERROR: POSCAR not found"
    echo "VASP calculation failed" > VASP_FAILED
    exit 1
fi

$VASP_CMD

EXIT_CODE=$?
echo "VASP exit code: $EXIT_CODE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "VASP calculation failed" > VASP_FAILED
    echo "ERROR: VASP failed with exit code $EXIT_CODE"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR 2>/dev/null
    exit 1
fi

if [ ! -f "vasprun.xml" ] || [ ! -s "vasprun.xml" ]; then
    echo "VASP calculation failed" > VASP_FAILED
    echo "ERROR: vasprun.xml missing or empty"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR 2>/dev/null
    exit 1
fi

# Verify vasprun.xml is valid
if ! grep -q "</modeling>" vasprun.xml; then
    echo "VASP calculation failed" > VASP_FAILED
    echo "ERROR: vasprun.xml is incomplete"
    rm -f CHGCAR CHG WAVECAR vasprun.xml WFULL AECCAR* TMPCAR 2>/dev/null
    exit 1
fi

touch VASP_DONE

# Cleanup
rm -f CHGCAR CHG WAVECAR WFULL AECCAR* TMPCAR 2>/dev/null

echo ""
echo "========================================"
echo "Displacement Complete"
echo "========================================"
echo "End time: $(date)"
