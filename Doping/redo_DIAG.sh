#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Optics/DIAG ; rm -v !("POSCAR"|"POTCAR"|"INCAR"|"KPOINTS"|"OUTCAR"|"log"|"vasprun.xml"|"sbp.sh") ; cd ../../.. ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp $line/Relax/CONTCAR $line/Optics/DIAG/POSCAR ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp $line/SC/WAVECAR $line/Optics/DIAG ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp INCAR_DIAG $line/Optics/DIAG/INCAR ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp sbp_DIAG.sh $line/Optics/DIAG/sbp.sh ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Optics/DIAG ; sbatch sbp.sh ; cd ../../../ ; done < "$input"
