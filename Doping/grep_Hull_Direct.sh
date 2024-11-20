#!/bin/bash
N_dir="$(grep POSCAR ../aflow_sym/uniq_poscar_list | tail -1 | awk '{print $1 }')"
for i in $(seq 1 ${N_dir}) ; do cd ${i}/Band ; vaspkit -task 211 > vaspkit.out ; cd ../.. ; done
for i in $(seq 1 ${N_dir}) ; do grep -l "Direct" ${i}/Band/BAND_GAP ; done | awk -F '/' '{print $1 }' > Direct_dir
input="./Direct_dir" ; while IFS= read -r line ; do same_sub=$(grep -w "$line" ../aflow_sym/uniq_poscar_list | awk '{print $2}' | awk -F '_' '{print $1"_"$2}') ; grep "$same_sub" ../aflow_sym/uniq_poscar_list ; done < "$input" | awk '{print $1}' > E_dir
input="./E_dir" ; while IFS= read -r line ; do e_cmd=$(grep "free energy    TOTEN  =" ${line}/Relax/OUTCAR | tail -1 | awk '{print $5}') d_cmd=$(grep -w "$line" ../aflow_sym/uniq_poscar_list | awk '{print $2}' | awk -F '_' '{print $2}') ; echo $d_cmd"_"$line $e_cmd ; done < "$input" | awk '!a[$0]++' > energy_list

