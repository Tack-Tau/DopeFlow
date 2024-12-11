#!/bin/bash

# Input log file
log_file="phonon-pp.log"

# Base calculation directory
calc_dir=$(pwd)

# Output file to store resubmission commands
resubmit_script="resubmit_phon_jobs.sh"
echo "#!/bin/bash" > "$resubmit_script"
echo "shopt -s extglob" >> "$resubmit_script"

# Parse the log file for errors
while IFS= read -r line; do
    if [[ "$line" =~ ^Error: ]]; then
        # Extract the structure directory and phonon directories with issues
        if [[ "$line" =~ Slurm\ errors\ detected\ in\ the\ following\ phonon\ directories\ for\ ([0-9]+):\ ([0-9]+) ]]; then
            struct_dir="${BASH_REMATCH[1]}"
            phonon_dir="${BASH_REMATCH[2]}"
            echo "Detected Slurm errors in structure directory: $struct_dir, phonon directory: $phonon_dir"

            # Prepare resubmission command
            echo "
            mkdir -p $calc_dir/$struct_dir/PHON/$phonon_dir
            cd $calc_dir/$struct_dir/PHON/$phonon_dir || exit
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cp $calc_dir/$struct_dir/PHON/POSCAR-$phonon_dir $calc_dir/$struct_dir/PHON/$phonon_dir/POSCAR
            cp $calc_dir/$struct_dir/PHON/{INCAR,POTCAR,sbp.sh} $calc_dir/$struct_dir/PHON/$phonon_dir/
            sbatch sbp.sh
            cd $calc_dir/$struct_dir/PHON/ || exit
            " >> "$resubmit_script"

        elif [[ "$line" =~ Missing\ or\ empty\ vasprun.xml\ in\ phonon\ directory\ ([0-9]+)\ for\ ([0-9]+) ]]; then
            phonon_dir="${BASH_REMATCH[1]}"
            struct_dir="${BASH_REMATCH[2]}"
            echo "Detected missing or empty vasprun.xml in structure directory: $struct_dir, phonon directory: $phonon_dir"

            # Prepare resubmission command
            echo "
            mkdir -p $calc_dir/$struct_dir/PHON/$phonon_dir
            cd $calc_dir/$struct_dir/PHON/$phonon_dir || exit
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cp $calc_dir/$struct_dir/PHON/POSCAR-$phonon_dir $calc_dir/$struct_dir/PHON/$phonon_dir/POSCAR
            cp $calc_dir/$struct_dir/PHON/{INCAR,POTCAR,sbp.sh} $calc_dir/$struct_dir/PHON/$phonon_dir/
            sbatch sbp.sh
            cd $calc_dir/$struct_dir/PHON/ || exit
            " >> "$resubmit_script"
        fi
    fi
done < "$log_file"

# Make the resubmit script executable
chmod +x "$resubmit_script"
echo "Resubmission script generated: $resubmit_script"

