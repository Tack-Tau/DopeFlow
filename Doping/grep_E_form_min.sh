#!/bin/bash
rm E_form_min_list.dat sorted_E_form_min_list.dat 2> /dev/null
Doping_dir="$(pwd)"
N_dir=$( cat -n ../aflow_sym/uniq_poscar_list | tail -1 | awk -F ' ' '{print $1}' )

function exists_in_list() {
  LIST=$1
  DELIMITER=$2
  VALUE=$3
  echo $LIST | tr "$DELIMITER" '\n' | grep -F -q -x "$VALUE"
}

calculate() { printf "%.6s\n" "$@" | bc -l; }

diverge_dir="$(input="./diverge_structs" ; while IFS= read -r line ; do diverge_dir=$(echo "$line" | awk '{print $1}') ; printf "%s " "$diverge_dir"  ; done < "$input")"

# Atomic energies
E_Na=-1.315661
E_Si=-5.423011
E_Ge=-4.487014

for i in $(seq 1 "${N_dir}") ; do
  if exists_in_list "$diverge_dir" " " $i ; then
    :
  else
    if grep -q "enthalpy is  TOTEN    =" $i/Relax/OUTCAR ; then
      e_final="$( echo "scale=6; $(grep "enthalpy is  TOTEN    =" $i/Relax/OUTCAR | tail -1 | awk '{printf "%.6f", $5 }') / $(grep "NIONS =" $i/Relax/OUTCAR | awk '{printf $12}' ) " | bc -l )"
    else
      e_final="$( echo "scale=6; $(grep "free energy    TOTEN  =" $i/Relax/OUTCAR | tail -1 | awk '{printf "%.6f", $5 }') / $(grep "NIONS =" $i/Relax/OUTCAR | awk '{printf $12}' ) " | bc -l )"
    fi

    # Extract element names and counts from POSCAR
    elem_line=$(sed -n '6p' $i/Relax/POSCAR)
    count_line=$(sed -n '7p' $i/Relax/POSCAR)

    # Initialize variables for counts
    n_Ge=0
    n_Na=0
    n_Si=0

    # Loop through elements and assign counts
    for idx in {1..3}; do
      element=$(echo $elem_line | awk -v i=$idx '{print $i}')
      count=$(echo $count_line | awk -v i=$idx '{print $i}')

      if [ "$element" == "Ge" ]; then
        n_Ge=$count
      elif [ "$element" == "Na" ]; then
        n_Na=$count
      elif [ "$element" == "Si" ]; then
        n_Si=$count
      fi
    done

    # Calculate formation energy
    if [[ $n_Ge -gt 0 || $n_Na -gt 0 || $n_Si -gt 0 ]]; then
      e_form=$(echo "scale=6; ($e_final - $n_Na*$E_Na - $n_Si*$E_Si - $n_Ge*$E_Ge) / ($n_Na + $n_Si + $n_Ge)" | bc -l)
      echo $i $e_form
    fi
  fi
done > E_form_min_list.dat

cat E_form_min_list.dat | sort -gk 2 > sorted_E_form_min_list.dat

