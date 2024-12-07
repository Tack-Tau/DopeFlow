#!/bin/sh
#SBATCH --partition=main
#SBATCH --job-name=phonon-pp
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
ulimit -s unlimited
ulimit -s
export OMP_NUM_THREADS=1

bash post-proc_phonon.sh > phonon-pp.log
