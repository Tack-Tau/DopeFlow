#!/bin/bash
shopt -s extglob
N_dir="$(grep POSCAR ../aflow_sym/uniq_poscar_list | tail -1 | awk '{print $1 }')"
error_dir="$( for i in $(seq 1 ${N_dir}) ; do grep -l "JOB\|CANCELLED" ${i}/Band/slurm-* | awk -F '/' '{print $1 }' ; done 2> /dev/null )"
for i in $error_dir ; do cd ${i}/Band ; rm !("POSCAR"|"INCAR"|"POTCAR"|"sbp.sh") ; cd ../.. ; done 1> /dev/null
for i in $error_dir ; do cp ${i}/SC/POSCAR ${i}/Band ; done 1> /dev/null
for i in $error_dir ; do cp ${i}/SC/CHGCAR ${i}/Band ; done 1> /dev/null
for i in $error_dir ; do cp INCAR_Band ${i}/Band/INCAR ; done 1> /dev/null
for i in $error_dir ; do cp sbp_Band.sh ${i}/Band/sbp.sh ; done 1> /dev/null
for i in $error_dir ; do cd ${i}/Band ; vaspkit -task 303 > vaspkit.out ;  cd ../.. ; done 1> /dev/null
for i in $error_dir ; do cd ${i}/Band ; mv POSCAR bk_POSCAR ; cp PRIMCELL.vasp POSCAR ; cd ../.. ; done 1> /dev/null
for i in $error_dir ; do cd ${i}/Band ; cp KPATH.in KPOINTS ; sed -i "2 s/^   20/   40/" KPOINTS ; cd ../.. ; done 1> /dev/null
for i in $error_dir ; do cd ${i}/Band ; sbatch sbp.sh ; cd ../.. ; done
