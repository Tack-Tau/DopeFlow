#!/bin/bash
shopt -s extglob

calc_dir="$PWD"
error_log_file="$calc_dir/optics_error.log"

# Ensure the error log file exists and clear its contents
> "$error_log_file"

# Function to check for SLURM errors in completed jobs
check_slurm_errors() {
    local dir=$1
    local calc=$2
    error_count=0
    while IFS= read -r line; do
        if grep -q -E "JOB|CANCELLED|error|failed|connection to proxy|error waiting for event|process manager error" $line/Optics/$calc/slurm-* 2>/dev/null; then
            error_count=$((error_count+1))
            echo "$line $calc" >> "$error_log_file"
        fi
    done < "$dir"
    return $error_count
}

# Function to prepare directory for calculation
prepare_directory() {
    local calc_type=$1
    local struct_dir=$2

    # Convert to absolute path if it's not already
    if [[ "$struct_dir" != /* ]]; then
        struct_dir="$calc_dir/$struct_dir"
    fi

    mkdir -p "$struct_dir/Optics/$calc_type"
    cd "$struct_dir/Optics/$calc_type" || return 1

    rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null

    case $calc_type in
        "SC")
            cp "$struct_dir/Relax/CONTCAR" ./POSCAR
            cp "$calc_dir/POTCAR_GW" ./POTCAR
            cp "$calc_dir/INCAR_SC" ./INCAR
            cp "$calc_dir/sbp_SC.sh" ./sbp.sh
            ;;
        "DIAG")
            cp "$struct_dir/Relax/CONTCAR" ./POSCAR
            cp "$calc_dir/POTCAR_GW" ./POTCAR
            cp "../SC/WAVECAR" ./
            cp "$calc_dir/INCAR_DIAG" ./INCAR
            cp "$calc_dir/sbp_DIAG.sh" ./sbp.sh
            ;;
        "GW0")
            cp "$struct_dir/Relax/CONTCAR" ./POSCAR
            cp "$calc_dir/POTCAR_GW" ./POTCAR
            cp "../DIAG/WAVECAR" ./
            cp "../DIAG/WAVEDER" ./
            cp "$calc_dir/INCAR_GW0" ./INCAR
            cp "$calc_dir/sbp_GW0.sh" ./sbp.sh
            ;;
        "BSE")
            cp "$struct_dir/Relax/CONTCAR" ./POSCAR
            cp "$calc_dir/POTCAR_GW" ./POTCAR
            cp "../GW0/"*.tmp ./
            cp "../GW0/WAVECAR" ./
            cp "../DIAG/WAVEDER" ./
            cp "$calc_dir/INCAR_BSE" ./INCAR
            cp "$calc_dir/sbp_BSE.sh" ./sbp.sh
            ;;
    esac
    cd "$calc_dir" || exit
}

# Function to submit and monitor a single job
submit_and_monitor() {
    local struct_dir=$1
    local calc_type=$2

    # Check current number of nodes in use
    sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
    
    # Wait if node usage is full
    while [ $sq_results -ge 60 ]; do
        sleep 30
        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
    done

    if ! prepare_directory "$calc_type" "$struct_dir"; then
        return 1
    fi

    cd "$struct_dir/Optics/$calc_type" || return 1
    
    slurm_output=$(sbatch sbp.sh)
    if [ $? -ne 0 ]; then
        echo "Failed to submit job for $calc_type calculation in $struct_dir" >> "$error_log_file"
        return 1
    fi

    job_id=$(echo $slurm_output | awk '{print $4}')

    # Monitor job completion
    while true; do
        if ! squeue -j "$job_id" &>/dev/null; then
            if grep -q -E "JOB|CANCELLED|error|failed" "$struct_dir/Optics/$calc_type/slurm-${job_id}.out"; then
                echo "Error in $calc_type calculation for $struct_dir" >> "$error_log_file"
                return 1
            fi
            break
        fi
        sleep 300
    done

    cd "$calc_dir" || exit
    return 0
}

# Function to submit all SC jobs for failed structures
submit_all_SC() {
    declare -A job_ids

    # First submit all SC jobs
    echo "Submitting SC calculations for failed structures..."
    while IFS=' ' read -r struct_dir calc_type; do
        # Only process structures that had SC failures or subsequent calculation failures
        case $calc_type in
            "SC"|"DIAG"|"GW0"|"BSE")
                # Check current number of nodes in use
                sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
                
                # Wait if node usage is full
                while [ $sq_results -ge 60 ]; do
                    sleep 30
                    sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
                done

                if ! prepare_directory "SC" "$struct_dir"; then
                    continue
                fi

                cd "$struct_dir/Optics/SC" || continue
                
                slurm_output=$(sbatch sbp.sh)
                if [ $? -ne 0 ]; then
                    continue
                fi

                job_id=$(echo $slurm_output | awk '{print $4}')
                job_ids[$struct_dir]=$job_id
                cd "$calc_dir" || exit
                ;;
        esac
    done < "$error_log_file"

    # Then monitor all SC jobs and submit subsequent calculations
    echo "Monitoring SC calculations..."
    local completed=0
    while [ $completed -lt ${#job_ids[@]} ]; do
        for struct_dir in "${!job_ids[@]}"; do
            job_id=${job_ids[$struct_dir]}
            
            # Skip if we've already processed this job
            [ "$job_id" == "done" ] && continue
            
            if ! squeue -j "$job_id" &>/dev/null; then
                if grep -q -E "JOB|CANCELLED|error|failed" "$struct_dir/Optics/SC/slurm-${job_id}.out"; then
                    echo "Error in SC calculation for $struct_dir" >> "$error_log_file"
                else
                    # If SC completed successfully, submit DIAG
                    submit_and_monitor "$struct_dir" "DIAG"
                fi
                job_ids[$struct_dir]="done"
                ((completed++))
            fi
        done
        sleep 300
    done
}

# First check for errors in all calculation types
for calc_type in SC DIAG GW0 BSE; do
    echo "Checking for errors in $calc_type calculations..."
    check_slurm_errors "$calc_dir/Direct_dir" "$calc_type"
done

# If no errors were found, exit
if [ ! -s "$error_log_file" ]; then
    echo "No errors found in any calculations."
    exit 0
fi

# Start resubmission with SC calculations
echo "Starting resubmission of failed calculations..."
submit_all_SC

echo "All calculations resubmitted and completed"