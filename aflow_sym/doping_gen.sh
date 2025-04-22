#!/bin/sh
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
# module use /projects/community/modulefiles
# module load intel/17.0.4 python/3.8.5-gc563
# export OMP_NUM_THREADS=1

ulimit -s unlimited
ulimit -s

source ~/.bashrc
conda activate shape

rm -r test_* uniq_poscar_list doping_log 2> /dev/null
echo -e "Si\nGe\n5\n500\ny" | python3 fp_cluster_doping.py > doping_log
