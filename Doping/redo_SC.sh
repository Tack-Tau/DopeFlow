#!/bin/bash
shopt -s extglob
N_dir="$(grep POSCAR ../aflow_sym/uniq_poscar_list | tail -1 | awk '{print $1 }')"
error_dir="$( for i in $(seq 1 ${N_dir}) ; do grep -l "JOB\|CANCELLED" ${i}/SC/slurm-* | awk -F '/' '{print $1 }' ; done 2> /dev/null )"
for i in $error_dir ; do cd ${i}/SC ; rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") ; cd ../../ ;  done 1> /dev/null
for i in $error_dir ; do cp ${i}/Relax/CONTCAR ${i}/SC/POSCAR ;  done 1> /dev/null
for i in $error_dir ; do cp INCAR_SC ${i}/SC/INCAR ;  done 1> /dev/null
for i in $error_dir ; do cp sbp_SC.sh ${i}/SC/sbp.sh ;  done 1> /dev/null
for i in $error_dir ; do cd ${i}/SC ; sbatch sbp.sh ; cd ../../ ; done
