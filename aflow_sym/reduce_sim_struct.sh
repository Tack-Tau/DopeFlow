#!/bin/bash
rm log uniq_poscar_list 2> /dev/null
imax=$(ls | grep POSCAR_ | sort -n -t _ -k 2 | tail -1 | awk -F '_' '{print $2 }')
for i in $(seq 1 ${imax}) ; do mkdir test_${i} ; done
for i in $(seq 1 ${imax}) ; do cp POSCAR_${i}_* test_${i} ; done
for i in $(seq 1 ${imax}) ; do aflow --compare_materials --np=8 -D ./test_${i} ; done 1> /dev/null
for i in $(seq 1 ${imax}) ; do grep "prototype=" test_${i}/material_comparison_output.out | awk -F '/' '{print $3 }' | sort -n -t _ -k 2 ; done >> uniq_poscar_list
