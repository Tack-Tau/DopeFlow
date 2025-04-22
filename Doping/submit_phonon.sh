#!/bin/bash

shopt -s extglob

# Define the calculation directory
calc_dir="$PWD"
# Log file for tracking submitted jobs
LOG_FILE="$calc_dir/job_PHON.log"

# Ensure the log file is created before anything else
if [[ ! -f "$LOG_FILE" ]]; then
    touch "$LOG_FILE"
fi

# Define a function to find and collect numbers from filenames in a specific structure directory
find_phonon_dir() {
    local struct_dir="$1"
    local phon_dir=""
    for file in "$struct_dir"/PHON/POSCAR-*; do
        if [[ -f "$file" ]]; then  # Check if it's a regular file
            num=$(basename "$file" | grep -oE '[0-9]+')  # Extract the number from the filename
            if [[ ! -z "$num" ]]; then
                phon_dir+="$num "  # Append the number to phon_dir
            fi
        fi
    done
    echo "$phon_dir"  # Output the collected numbers as a space-separated string
}

# Function to check if a phon_dir has already been submitted for a specific structure directory
is_job_submitted() {
    local phon_dir="$1"
    local struct_dir="$2"
    grep -q "Submitted job for $phon_dir in structure directory: $struct_dir" "$LOG_FILE"
}

# Function to retrieve the last submitted phon_dir, total phonon count, and struct_dir pair from the log file
get_last_submitted_job() {
    grep "Submitted job for" "$LOG_FILE" | tail -n 1 | awk -F ' ' '{print $4, $6, $10}'
}

# Submit jobs in batches of 10 when the queue is not full
submit_in_batches() {
    local job_counter=0
    local total_phon_dirs=($1)
    local total_count=${#total_phon_dirs[@]}
    local struct_dir="$2"

    # Retrieve the last submitted phonon directory, total count, and structure directory, if any
    last_job_info=$(get_last_submitted_job)
    last_submitted_phon_dir=""
    last_submitted_tot_phon_count=""
    last_submitted_struct_dir=""

    if [[ ! -z "$last_job_info" ]]; then
        last_submitted_phon_dir=$(echo "$last_job_info" | awk '{print $1}')
        last_submitted_tot_phon_count=$(echo "$last_job_info" | awk '{print $2}')
        last_submitted_struct_dir=$(echo "$last_job_info" | awk '{print $3}')
    fi

    # Check if resuming submission for the current structure directory
    if [[ "$struct_dir" == "$last_submitted_struct_dir" ]]; then
        # Compare as strings, not numbers
        if [[ "$last_submitted_phon_dir" != "" ]]; then
            # Find the index of the last submitted phonon directory in the list
            while [[ $job_counter -lt $total_count && "${total_phon_dirs[$job_counter]}" != "$last_submitted_phon_dir" ]]; do
                ((job_counter++))
            done
            ((job_counter++))  # Start from the next phonon directory
        fi
    else
        job_counter=0  # Start from the beginning of the list for new structure directory
    fi

    # Submit jobs in batches of 10
    while [[ $job_counter -lt $total_count ]]; do
        sq_results=$(squeue -u $USER | sed 1d | wc -l)

        # Submit jobs only if running jobs are <= 50
        if [[ $sq_results -le 50 ]]; then
            for (( i=0; i<10 && job_counter<total_count; i++ )); do
                phon_dir="${total_phon_dirs[$job_counter]}"

                # Check if this job has already been submitted for this structure directory
                if is_job_submitted "$phon_dir" "$struct_dir"; then
                    echo "Skipping already submitted job for $phon_dir in structure directory: $struct_dir"
                    ((job_counter++))
                else
                    mkdir -p "$calc_dir/$struct_dir/PHON/$phon_dir"
                    cp "$calc_dir/$struct_dir/PHON/POSCAR-$phon_dir" "$calc_dir/$struct_dir/PHON/$phon_dir/POSCAR"
                    cp "$calc_dir/$struct_dir/PHON/INCAR" "$calc_dir/$struct_dir/PHON/POTCAR" "$calc_dir/$struct_dir/PHON/sbp.sh" "$calc_dir/$struct_dir/PHON/$phon_dir/"
                    cd "$calc_dir/$struct_dir/PHON/$phon_dir" || exit
                    sbatch sbp.sh
                    cd "$calc_dir/$struct_dir/PHON/" || exit

                    # Log the submission with structure directory and phon_dir
                    echo "Submitted job for $phon_dir / $total_count in structure directory: $struct_dir" >> "$LOG_FILE"
                    ((job_counter++))
                fi
            done
        else
            echo "Waiting for jobs to finish... ($sq_results jobs in the queue)"
            sleep 600  # Wait for 10 minutes before rechecking the queue
        fi
    done
}

# Check if the log file is empty and initialize fresh start if so
if [[ ! -s "$LOG_FILE" ]]; then
    echo "Log file is empty. Initializing fresh start for all directories in phonon_list."
    for i in $(cat phonon_list); do
        echo "Processing structure directory: $i"
        mkdir -p "$calc_dir/${i}/PHON"
        cp "$calc_dir/${i}/Relax/CONTCAR" "$calc_dir/${i}/PHON/POSCAR"
        cd "$calc_dir/${i}/PHON" || exit
        vaspkit -task 602 1> /dev/null
        cp POSCAR bk_POSCAR
        cp PRIMCELL.vasp POSCAR
        vaspkit -task 303 1> /dev/null
        
        # Copy necessary files for job submission
        cp "$calc_dir/INCAR_PHON" INCAR
        cp "$calc_dir/sbp_PHON.sh" sbp.sh
        cp "$calc_dir/"{convert_kpath.sh,generate_supercell.sh,extract_band_conf.sh,preprocess_high_symmetry_points.sh} ./
        
        # Generate supercells if needed
        bash generate_supercell.sh
        
        # Collect phonon directories for this structure directory
        phon_dirs=$(find_phonon_dir "$calc_dir/${i}")
        
        # Submit jobs in batches for this structure directory
        cd "$calc_dir" || exit
        submit_in_batches "$phon_dirs" "$i"
        
        echo "Finished submission for structure directory: $i"
    done
    echo "Fresh start initialization complete."
    exit 0
fi

# Retrieve the last submitted job
last_job_info=$(get_last_submitted_job)
last_submitted_struct_dir=""
last_submitted_phon_dir=""
last_submitted_tot_phon_count=""

if [[ ! -z "$last_job_info" ]]; then
    last_submitted_phon_dir=$(echo "$last_job_info" | awk '{print $1}')
    last_submitted_tot_phon_count=$(echo "$last_job_info" | awk '{print $2}')
    last_submitted_struct_dir=$(echo "$last_job_info" | awk '{print $3}')
fi

# Preprocessing if the log file is not empty, only process the next unsubmitted struct_dir
if [[ -s "$LOG_FILE" ]]; then
    skip_to_next_struct=false
    next_struct=""
    for i in $(cat phonon_list); do
        # Find the next unsubmitted structure after the last submitted one
        if [[ "$skip_to_next_struct" == true ]]; then
            next_struct="$i"
            break  # Exit loop after finding the next structure
        fi
        
        # Mark when we've found the last submitted structure
        if [[ "$i" == "$last_submitted_struct_dir" ]]; then
            skip_to_next_struct=true
        fi
    done

    # Only preprocess the next structure if found
    if [[ ! -z "$next_struct" ]]; then
        echo "Processing structure directory: $next_struct"
        mkdir -p "$calc_dir/${next_struct}/PHON"
        cp "$calc_dir/${next_struct}/Relax/CONTCAR" "$calc_dir/${next_struct}/PHON/POSCAR"
        cd "$calc_dir/${next_struct}/PHON" || exit
        vaspkit -task 602 1> /dev/null
        cp POSCAR bk_POSCAR
        cp PRIMCELL.vasp POSCAR
        vaspkit -task 303 1> /dev/null
        
        # Copy necessary files and generate supercells
        cp "$calc_dir/INCAR_PHON" INCAR
        cp "$calc_dir/sbp_PHON.sh" sbp.sh
        cp "$calc_dir/"{convert_kpath.sh,generate_supercell.sh,extract_band_conf.sh,preprocess_high_symmetry_points.sh} ./
        bash generate_supercell.sh
        
        cd "$calc_dir" || exit
    fi
fi

# Skip fully submitted structure directories before the last submitted one
struct_started=false
for i in $(cat phonon_list); do
    # Skip structure directories before the last submitted one
    if [[ "$i" == "$last_submitted_struct_dir" ]]; then
        struct_started=true
    fi
    if [[ "$struct_started" == false ]]; then
        continue  # Skip structure directories until reaching the last one
    fi

    # Collect phonon directories for this structure directory
    phon_dirs=$(find_phonon_dir "$calc_dir/${i}")

    # If we're resuming, skip already submitted phonon dirs for the current struct_dir
    job_counter=0
    if [[ "$i" == "$last_submitted_struct_dir" ]]; then
        while [[ $job_counter -lt ${#phon_dirs[@]} && "${phon_dirs[$job_counter]}" != "$last_submitted_phon_dir" ]]; do
            ((job_counter++))
        done
        ((job_counter++))  # Start from the next phonon dir after the last submitted one
    fi

    # If there are no phonon directories, prepare a fresh start
    if [[ -z "$phon_dirs" ]]; then
        echo "No phonon directories found for $i. Preparing fresh start."
        # Create PHON directory if it doesn't exist
        mkdir -p "$calc_dir/$i/PHON"
        
        # Check and copy CONTCAR from Relax if POSCAR doesn't exist in PHON
        if [[ ! -f "$calc_dir/$i/PHON/POSCAR" && -f "$calc_dir/$i/Relax/CONTCAR" ]]; then
            cp "$calc_dir/$i/Relax/CONTCAR" "$calc_dir/$i/PHON/POSCAR"
        fi
        
        # Navigate to PHON directory
        cd "$calc_dir/$i/PHON" || exit
        
        # Run vaspkit tasks if not already done
        if [[ ! -f "$calc_dir/$i/PHON/PRIMCELL.vasp" ]]; then
            vaspkit -task 602 1> /dev/null
            cp POSCAR bk_POSCAR
            cp PRIMCELL.vasp POSCAR
            vaspkit -task 303 1> /dev/null
        fi
        
        # Copy necessary script files if they don't exist
        if [[ ! -f "$calc_dir/$i/PHON/generate_supercell.sh" ]]; then
            cp "$calc_dir/INCAR_PHON" INCAR
            cp "$calc_dir/sbp_PHON.sh" sbp.sh
            cp "$calc_dir/"{convert_kpath.sh,generate_supercell.sh,extract_band_conf.sh,preprocess_high_symmetry_points.sh} ./
        fi
        
        # Generate supercells
        bash generate_supercell.sh
        phon_dirs=$(find_phonon_dir "$calc_dir/${i}")  # Update the phon_dirs after generating supercells
    fi

    # Copy necessary files for job submission
    cp "$calc_dir/INCAR_PHON" "$calc_dir/${i}/PHON/INCAR"
    cp "$calc_dir/sbp_PHON.sh" "$calc_dir/${i}/PHON/sbp.sh"
    cp "$calc_dir/"{redo_phonon.sh,convert_kpath.sh,generate_supercell.sh,extract_band_conf.sh,preprocess_high_symmetry_points.sh} "$calc_dir/${i}/PHON/"

    # Submit jobs in batches for this structure directory $i
    submit_in_batches "$phon_dirs" "$i"

    echo "Finished submission for structure directory: $i"
    cd "$calc_dir"
done

echo "All structure directories in phonon_list have been processed."
