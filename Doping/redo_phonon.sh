#!/bin/bash

shopt -s extglob

# Define a function to find and collect numbers from filenames
find_phonon_dir() {
    local phon_dir=""
    for file in POSCAR-*; do
        if [[ -f "$file" ]]; then  # Check if it's a regular file
            num=$(echo "$file" | grep -oE '[0-9]+')  # Extract the number from the filename
            if [[ ! -z "$num" ]]; then
                phon_dir+="$num "  # Append the number to phon_dir
            fi
        fi
    done
    echo "$phon_dir"  # Output the collected numbers as a space-separated string
}

error_dir=$(for phon_dir in $(find_phonon_dir); do
        error_phon=$(grep -l "error" "$phon_dir"/slurm-* 2>/dev/null | awk -F '/' '{print $1}')
        if [[ -n "$error_phon" ]]; then
            echo "$error_phon"
        fi
    done)


for i in $error_dir ; do cd ${i}/ ; rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") ; cd .. ; done 2> /dev/null
for i in $error_dir ; do cp INCAR ${i}/INCAR ; done
for i in $error_dir ; do cp sbp.sh ${i}/sbp.sh ; done
for i in $error_dir ; do cd ${i}/ ; sbatch sbp.sh ; cd .. ; done
