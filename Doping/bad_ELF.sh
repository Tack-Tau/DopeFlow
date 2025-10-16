#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do grep -l "BAD TERMINATION" $line/ELF/log ; done < "$input" 2> /dev/null | awk -F '/' '{print $1 }' > error_dir
input="./error_dir" ; while IFS= read -r line ; do cd $line/ELF ; rm -v !("POSCAR"|"POTCAR"|"INCAR"|"OUTCAR"|"log"|"vasprun.xml"|"sbp.sh") ; cd ../../ ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/Relax/CONTCAR $line/ELF/POSCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp $line/SC/CHGCAR $line/ELF/ ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp INCAR_ELF $line/ELF/INCAR ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cp sbp_ELF.sh $line/ELF/sbp.sh ; done < "$input" 1> /dev/null
input="./error_dir" ; while IFS= read -r line ; do cd $line/ELF ; sbatch sbp.sh ; cd ../../ ; done < "$input"

