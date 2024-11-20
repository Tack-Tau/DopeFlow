#!/bin/sh
#SBATCH --partition=main
#SBATCH --job-name=phonon
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load intel/17.0.4
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

/home/$USER/apps/writekp.py 0.07
mpirun -n 64 /home/$USER/apps/vasp.5.4.4_intel/bin/vasp_std > log
