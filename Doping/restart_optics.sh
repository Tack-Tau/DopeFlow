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
            slurm_file="$calc_dir/$struct_dir/Optics/$calc_type/slurm-${job_id}.out"
            if [[ -f "$slurm_file" ]] && grep -q -E "JOB|CANCELLED|error|failed" "$slurm_file"; then
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

# Function to check if a job completed successfully
job_completed_successfully() {
    local job_id=$1
    local struct_dir=$2
    local calc_type=$3
    local timestamp_file="/tmp/last_job_check_${struct_dir}_${calc_type}"
    
    # Check if job is still in queue using grep
    if squeue -u $USER | grep -q "$job_id"; then
        # Only log every 15 minutes (900 seconds)
        current_time=$(date +%s)
        if [ ! -f "$timestamp_file" ] || [ $((current_time - $(cat "$timestamp_file"))) -ge 900 ]; then
            echo "Job $job_id still running for $struct_dir ($calc_type)" >> "$log_file"
            echo "$current_time" > "$timestamp_file"
        fi
        return 1  # Job is still running
    fi

    # Clean up timestamp file when job is done
    rm -f "$timestamp_file"

    # Check OUTCAR & slurm output file for completion
    local slurm_file="$struct_dir/Optics/$calc_type/slurm-${job_id}.out"
    local outcar_file="$struct_dir/Optics/$calc_type/OUTCAR"
    echo "Checking job completion status from slurm file: $slurm_file" >> "$log_file"
    
    if [[ -f "$slurm_file" ]]; then
        # Check for common SLURM/VASP error patterns
        if grep -q -E "CANCELLED|error|failed|BAD TERMINATION|SIGSEGV|segmentation fault occurred" "$slurm_file"; then
            echo "Error found in job $job_id for $struct_dir ($calc_type)" >> "$error_log_file"
            return 1
        fi
        
        # Check for VASP specific completion markers in OUTCAR
        if [[ -f "$outcar_file" ]] && grep -q "General timing and accounting informations for this job:" "$outcar_file"; then
            
            case $calc_type in
                "SC")
                    # Check if WAVECAR exists and is not empty
                    if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]]; then
                        echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$log_file"
                        return 0
                    fi
                    ;;
                "DIAG")
                    # Check if both WAVECAR and WAVEDER exist
                    if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                       [[ -s "$struct_dir/Optics/$calc_type/WAVEDER" ]]; then
                        echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$log_file"
                        return 0
                    fi
                    ;;
                "GW0")
                    # Check for GW0 specific output files
                    if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                       ls "$struct_dir/Optics/$calc_type/"*.tmp 1> /dev/null 2>&1; then
                        echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$log_file"
                        return 0
                    fi
                    ;;
                "BSE")
                    # Check for BSE specific output files
                    if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                       [[ -s "$struct_dir/Optics/$calc_type/WAVEDER" ]]; then
                        echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$log_file"
                        return 0
                    fi
                    ;;
            esac
        fi
    else
        echo "Slurm output file not found: $slurm_file" >> "$error_log_file"
    fi

    echo "Job $job_id completion check failed for $struct_dir ($calc_type)" >> "$log_file"
    return 1  # Default to failure if we can't confirm success
}

# Function to submit all SC jobs first
submit_all_SC() {
    declare -A job_ids  # Associate array to store job IDs for each structure
    declare -A calc_stage  # Track which calculation stage each structure is at
    local status_timestamp_file="/tmp/last_status_print"
    local status_interval=900  # 15 minutes in seconds

    # Initialize all structures to SC stage
    while IFS= read -r struct_dir; do
        calc_stage[$struct_dir]="SC"
    done < "$calc_dir/Direct_dir"

    # Submit all initial SC jobs
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
        if [ $? -eq 0 ]; then
            new_job_id=$(echo $slurm_output | awk '{print $4}')
            job_ids[$struct_dir]=$new_job_id
            calc_stage[$struct_dir]="SC"
            echo "Submitted SC job for $struct_dir with ID $new_job_id" >> "$log_file"
            all_complete=0
        else
            echo "Failed to submit SC job for $struct_dir" >> "$error_log_file"
        fi
        cd "$calc_dir" || exit
    done < "$calc_dir/Direct_dir"

    # Monitor jobs and chain calculations
    local all_complete=0
    while [ $all_complete -eq 0 ]; do
        all_complete=1
        
        for struct_dir in "${!calc_stage[@]}"; do
            current_job_id=${job_ids[$struct_dir]}
            current_stage=${calc_stage[$struct_dir]}
            
            case $current_stage in
                "SC")
                    if job_completed_successfully "$current_job_id" "$struct_dir" "SC"; then
                        echo "SC calculation completed for $struct_dir, preparing DIAG calculation" >> "$log_file"
                        if prepare_directory "DIAG" "$struct_dir"; then
                            cd "$struct_dir/Optics/DIAG" || continue
                            slurm_output=$(sbatch sbp.sh)
                            if [ $? -eq 0 ]; then
                                new_job_id=$(echo $slurm_output | awk '{print $4}')
                                job_ids[$struct_dir]=$new_job_id
                                calc_stage[$struct_dir]="DIAG"
                                echo "Submitted DIAG job for $struct_dir with ID $new_job_id" >> "$log_file"
                                all_complete=0
                            fi
                            cd "$calc_dir" || exit
                        fi
                    else
                        all_complete=0
                    fi
                    ;;
                "DIAG")
                    if job_completed_successfully "$current_job_id" "$struct_dir" "DIAG"; then
                        echo "DIAG calculation completed for $struct_dir, preparing GW0 calculation" >> "$log_file"
                        if prepare_directory "GW0" "$struct_dir"; then
                            cd "$struct_dir/Optics/GW0" || continue
                            slurm_output=$(sbatch sbp.sh)
                            if [ $? -eq 0 ]; then
                                new_job_id=$(echo $slurm_output | awk '{print $4}')
                                job_ids[$struct_dir]=$new_job_id
                                calc_stage[$struct_dir]="GW0"
                                echo "Submitted GW0 job for $struct_dir with ID $new_job_id" >> "$log_file"
                                all_complete=0
                            fi
                            cd "$calc_dir" || exit
                        fi
                    else
                        all_complete=0
                    fi
                    ;;
                "GW0")
                    if job_completed_successfully "$current_job_id" "$struct_dir" "GW0"; then
                        echo "GW0 calculation completed for $struct_dir, preparing BSE calculation" >> "$log_file"
                        if prepare_directory "BSE" "$struct_dir"; then
                            cd "$struct_dir/Optics/BSE" || continue
                            slurm_output=$(sbatch sbp.sh)
                            if [ $? -eq 0 ]; then
                                new_job_id=$(echo $slurm_output | awk '{print $4}')
                                job_ids[$struct_dir]=$new_job_id
                                calc_stage[$struct_dir]="BSE"
                                echo "Submitted BSE job for $struct_dir with ID $new_job_id" >> "$log_file"
                                all_complete=0
                            fi
                            cd "$calc_dir" || exit
                        fi
                    else
                        all_complete=0
                    fi
                    ;;
                "BSE")
                    if job_completed_successfully "$current_job_id" "$struct_dir" "BSE"; then
                        echo "BSE calculation completed for $struct_dir" >> "$log_file"
                        calc_stage[$struct_dir]="COMPLETE"
                    else
                        all_complete=0
                    fi
                    ;;
            esac
        done
        
        # Print status update every 15 minutes
        current_time=$(date +%s)
        if [ ! -f "$status_timestamp_file" ] || [ $((current_time - $(cat "$status_timestamp_file"))) -ge $status_interval ]; then
            echo "Current status at $(date):" >> "$log_file"
            for struct_dir in "${!calc_stage[@]}"; do
                echo "$struct_dir: ${calc_stage[$struct_dir]} (Job ID: ${job_ids[$struct_dir]})" >> "$log_file"
            done
            echo "$current_time" > "$status_timestamp_file"
        fi
        
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