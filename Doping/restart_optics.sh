#!/bin/bash
shopt -s extglob

calc_dir="$PWD"
log_file="$calc_dir/restart_optical_jobs.log"
error_log_file="$calc_dir/optics_error.log"

# Function to cleanup old calculations
cleanup_old_calcs() {
    local struct_dir=$1
    for calc_type in SC DIAG GW0 BSE; do
        if [ -d "$struct_dir/Optics/$calc_type" ]; then
            cd "$struct_dir/Optics/$calc_type" || continue
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cd "$calc_dir" || exit
        fi
    done
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
    if ! cd "$struct_dir/Optics/$calc_type"; then
        echo "Error: Could not change to directory $struct_dir/Optics/$calc_type" >> "$error_log_file"
        return 1
    fi

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

    if ! prepare_directory "$calc_type" "$struct_dir"; then
        return 1
    fi

    # Check current number of nodes in use
    sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
    
    # Wait if node usage is full
    while [ $sq_results -ge 60 ]; do
        sleep 30
        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
    done

    cd "$struct_dir/Optics/$calc_type" || return 1
    
    slurm_output=$(sbatch sbp.sh)
    if [ $? -ne 0 ]; then
        echo "Failed to submit job for $calc_type calculation in $struct_dir" >> "$error_log_file"
        return 1
    fi

    job_id=$(echo $slurm_output | awk '{print $4}')
    echo "Submitted $calc_type job for $struct_dir with ID $job_id" >> "$log_file"

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

# Function to submit all SC jobs first
submit_all_SC() {
    declare -A job_ids  # Associate array to store job IDs for each structure

    # First submit all SC jobs
    echo "Submitting all SC calculations..."
    while IFS= read -r struct_dir; do
        # Check current number of nodes in use
        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        
        # Wait if node usage is full
        while [ $sq_results -ge 60 ]; do
            sleep 30
            sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        done

        if ! prepare_directory "SC" "$struct_dir"; then
            echo "Failed to prepare directory for SC calculation in $struct_dir" >> "$error_log_file"
            continue
        fi

        cd "$struct_dir/Optics/SC" || continue
        
        slurm_output=$(sbatch sbp.sh)
        if [ $? -ne 0 ]; then
            echo "Failed to submit job for SC calculation in $struct_dir" >> "$error_log_file"
            continue
        fi

        job_id=$(echo $slurm_output | awk '{print $4}')
        job_ids[$struct_dir]=$job_id
        echo "Submitted SC job for $struct_dir with ID $job_id" >> "$log_file"
        cd "$calc_dir" || exit
    done < "$calc_dir/Direct_dir"

    # Then monitor all SC jobs and submit subsequent calculations
    echo "Monitoring all SC calculations..."
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

# Initialize log files
echo "Starting restart of optical calculations at $(date)" > "$log_file"
echo "Starting error log at $(date)" > "$error_log_file"

# Cleanup old calculations
while IFS= read -r struct_dir; do
    cleanup_old_calcs "$struct_dir"
done < "$calc_dir/Direct_dir"

# Main execution
echo "Restarting all optical calculations from scratch..."
submit_all_SC

echo "All calculations completed or failed. Check logs for details." 