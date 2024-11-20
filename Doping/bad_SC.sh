#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do grep -l "BAD TERMINATION" $line/SC/log ; done < "$input" 2> /dev/null | awk -F '/' '{print $1 }' > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd $line/SC ; rm -v !("POSCAR"|"POTCAR"|"INCAR"|"KPOINTS"|"OUTCAR"|"log"|"vasprun.xml"|"sbp.sh") ; cd ../../ ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Relax/CONTCAR $line/SC/POSCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp INCAR_SC $line/SC/INCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp sbp_SC.sh $line/SC/sbp.sh ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cd $line/SC ; sbatch sbp.sh ; cd ../../ ; done < "$input"
