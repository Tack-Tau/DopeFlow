#!/bin/bash
mv uniq_poscar_list uniq_poscar_list.bk
cat -n uniq_poscar_list.bk | sort -gk 1 > uniq_poscar_list
rm uniq_poscar_list.bk
N_dir="$(grep POSCAR uniq_poscar_list | tail -1 | awk '{print $1 }')"
aflow_sym_dir="$(pwd)"
cd ../Doping
for i in $(seq 1 ${N_dir}) ; do mkdir ${i} ; done
cd $aflow_sym_dir
input="./uniq_poscar_list" ; while IFS= read -r line ; do read -ra arr_tmp -d '' <<<"${line}" ; cp ./"${arr_tmp[1]}" ../Doping/"${arr_tmp[0]}"/POSCAR ; done < "$input"
