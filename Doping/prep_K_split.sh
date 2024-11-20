#!/bin/bash
tail -n +5 KPATH.in > KPATH_tmp.in
head -n 4 KPATH.in > K_head
num_tmp=$(echo "$(wc -l < KPATH_tmp.in) / 3 " | bc)
bash split_K.sh KPATH_tmp.in $num_tmp
for i in $(seq 1 $num_tmp) ; do sed -i -f - KPATH.in.$i < <(sed 's/^/1i/' K_head) ; done
for i in $(seq 1 $num_tmp) ; do mkdir PBE0-$i ; done
for i in $(seq 1 $num_tmp) ; do cp KPATH.in.$i PBE0-$i/KPATH.in ; done
for i in $(seq 1 $num_tmp) ; do cp INCAR POSCAR POTCAR CHGCAR SYMMETRY HIGH_SYMMETRY_POINTS sbp.sh PBE0-$i ; done
