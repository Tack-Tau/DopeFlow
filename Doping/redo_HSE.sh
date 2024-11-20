#!/bin/bash
shopt -s extglob
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Fake_SC ; rm !("POSCAR"|"INCAR"|"POTCAR"|"sbp.sh") ; cd ../.. ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp $line/SC/POSCAR $line/Fake_SC ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp $line/SC/CHGCAR $line/Fake_SC ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp INCAR_HSE $line/Fake_SC/INCAR ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cp sbp_HSE.sh $line/Fake_SC/sbp.sh ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Fake_SC ; vaspkit -task 303 > vaspkit.out ;  cd ../.. ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Fake_SC ; mv POSCAR bk_POSCAR ; cp PRIMCELL.vasp POSCAR ; cd ../.. ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Fake_SC ; vaspkit -task 251 > vaspkit.out ;       cd ../.. ; done < "$input" 1> /dev/null
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Fake_SC ; sbatch sbp.sh ; cd ../.. ; done < "$input"
