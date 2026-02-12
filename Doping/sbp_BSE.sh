#!/bin/sh
#SBATCH --partition=main
#SBATCH --job-name=BSE
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=128GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load intel/17.0.4
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

/home/$USER/apps/writekp.py 0.04
mpirun -n 8 /home/$USER/apps/vasp.5.4.4_intel/bin/vasp_std > log
