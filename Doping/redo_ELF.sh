#!/bin/bash
shopt -s extglob
N_dir="$(grep POSCAR ../aflow_sym/uniq_poscar_list | tail -1 | awk '{print $1 }')"
error_dir="$( for i in $(seq 1 ${N_dir}) ; do grep -l "JOB\|CANCELLED" ${i}/ELF/slurm-* | awk -F '/' '{print $1 }' ; done 2> /dev/null )"
for i in $error_dir ; do cd ${i}/ELF ; rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") ; cd ../../ ; done 1> /dev/null
for i in $error_dir ; do cp ${i}/Relax/CONTCAR ${i}/ELF/POSCAR ; done 1> /dev/null
for i in $error_dir ; do cp ${i}/SC/CHGCAR ${i}/ELF/ ; done 1> /dev/null
for i in $error_dir ; do cp INCAR_ELF ${i}/ELF/INCAR ; done 1> /dev/null
for i in $error_dir ; do cp sbp_ELF.sh ${i}/ELF/sbp.sh ; done 1> /dev/null
for i in $error_dir ; do cd ${i}/ELF ; sbatch sbp.sh ; cd ../../ ; done

