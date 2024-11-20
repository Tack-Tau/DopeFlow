#!/bin/bash
N_dir="$(grep POSCAR ../aflow_sym/uniq_poscar_list | tail -1 | awk '{print $1 }')"
for i in $(seq 1 ${N_dir}) ; do cd ${i} ; mkdir Relax SC DOS Band ; cd .. ; done
for i in $(seq 1 ${N_dir}) ; do cp ${i}/POSCAR ${i}/Relax ; done
for i in $(seq 1 ${N_dir}) ; do cp POTCAR ${i}/Relax ; done
for i in $(seq 1 ${N_dir}) ; do cp POTCAR ${i}/SC ; done
for i in $(seq 1 ${N_dir}) ; do cp POTCAR ${i}/DOS ; done
for i in $(seq 1 ${N_dir}) ; do cp POTCAR ${i}/Band ; done
for i in $(seq 1 ${N_dir}) ; do cp INCAR_Relax ${i}/Relax/INCAR ; done
for i in $(seq 1 ${N_dir}) ; do cp sbp_Relax.sh ${i}/Relax/sbp.sh ; done
for i in $(seq 1 ${N_dir}) ; do cp INCAR_SC ${i}/SC/INCAR ; done
for i in $(seq 1 ${N_dir}) ; do cp sbp_SC.sh ${i}/SC/sbp.sh ; done
for i in $(seq 1 ${N_dir}) ; do cp INCAR_DOS ${i}/DOS/INCAR ; done
for i in $(seq 1 ${N_dir}) ; do cp sbp_DOS.sh ${i}/DOS/sbp.sh ; done
for i in $(seq 1 ${N_dir}) ; do cp INCAR_Band ${i}/Band/INCAR ; done
for i in $(seq 1 ${N_dir}) ; do cp sbp_Band.sh ${i}/Band/sbp.sh ; done
