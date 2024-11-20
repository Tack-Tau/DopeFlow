#!/bin/bash
rm diverge_structs 2> /dev/null
Doping_dir="$(pwd)"
N_dir=$( cat -n ../aflow_sym/uniq_poscar_list | tail -1 | awk -F ' ' '{print $1}' )

for i in $(seq 1 "${N_dir}"); do
  cd "$i/Relax" || continue
  if diff <(aflow --aflowSG < POSCAR) <(aflow --aflowSG < CONTCAR) > /dev/null; then
    :
  else
    dir=$(pwd | awk -F '/' '{print $(NF-1)}')
    sed "${dir}q;d" "$Doping_dir/../aflow_sym/uniq_poscar_list"
  fi
  cd ../..
done >> diverge_structs

