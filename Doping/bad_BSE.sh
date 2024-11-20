#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do grep -l "BAD TERMINATION" $line/Optics/BSE/log ; done < "$input" 2> /dev/null | awk -F '/' '{print $1 }' > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd $line/Optics/BSE ; rm -v !("POSCAR"|"POTCAR"|"INCAR"|"KPOINTS"|"OUTCAR"|"log"|"vasprun.xml"|"sbp.sh") ; cd ../../.. ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Optics/GW0/*.tmp $line/Optics/BSE ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Optics/GW0/WAVECAR $line/Optics/BSE ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Optics/DIAG/WAVEDER $line/Optics/BSE ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp INCAR_BSE $line/Optics/BSE/INCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp sbp_BSE.sh $line/Optics/BSE/sbp.sh ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cd $line/Optics/BSE ; sbatch sbp.sh ; cd ../../../ ; done < "$input"
