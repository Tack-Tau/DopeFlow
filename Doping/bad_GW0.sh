#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do grep -l "BAD TERMINATION" $line/Optics/GW0/log ; done < "$input" 2> /dev/null | awk -F '/' '{print $1 }' > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd $line/Optics/GW0 ; rm -v !("POSCAR"|"POTCAR"|"INCAR"|"KPOINTS"|"OUTCAR"|"log"|"vasprun.xml"|"sbp.sh") ; cd ../../.. ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Relax/CONTCAR  $line/Optics/GW0/POSCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Optics/DIAG/WAVECAR $line/Optics/GW0 ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Optics/DIAG/WAVEDER $line/Optics/GW0 ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp INCAR_GW0 $line/Optics/GW0/INCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp sbp_GW0.sh $line/Optics/GW0/sbp.sh ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cd $line/Optics/GW0 ; sbatch sbp.sh ; cd ../../../ ; done < "$input"
