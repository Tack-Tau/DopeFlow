#!/bin/sh
#SBATCH --partition=main
#SBATCH --job-name=GW0
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=96:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load intel/17.0.4
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

/home/$USER/apps/writekp.py 0.04
mpirun -n 16 /home/$USER/apps/vasp.5.4.4_intel/bin/vasp_std > log
