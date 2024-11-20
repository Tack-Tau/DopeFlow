#!/bin/sh
#SBATCH --partition=main
#SBATCH --job-name=PHONON
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=190GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load intel/17.0.4
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

echo -e "102\n2\n0.04\n"| vaspkit 1> /dev/null
srun --mpi=pmi2 /home/$USER/apps/vasp.5.4.4_intel/bin/vasp_std > log && rm CHGCAR WAVECAR
