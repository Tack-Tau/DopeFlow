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
    echo "Submitted $calc_type job for $struct_dir with ID $job_id" >> "$error_log_file"

    # Monitor job completion with improved status checking
    while true; do
        if job_completed_successfully "$job_id" "$struct_dir" "$calc_type"; then
            cd "$calc_dir" || exit
            return 0
        elif ! squeue -u $USER | grep -q "$job_id"; then
            # Job is no longer in queue and didn't complete successfully
            echo "Job $job_id failed for $struct_dir ($calc_type)" >> "$error_log_file"
            cd "$calc_dir" || exit
            return 1
        fi
        sleep 300
    done
}

# Function to submit calculations starting from a specific stage
submit_from_stage() {
    local struct_dir=$1
    local start_stage=$2
    local stages=("SC" "DIAG" "GW0" "BSE")
    local start_index=0

    # Find the index of the starting stage
    for i in "${!stages[@]}"; do
        if [[ "${stages[$i]}" == "$start_stage" ]]; then
            start_index=$i
            break
        fi
    done

    # Submit jobs sequentially from the starting stage
    for ((i=start_index; i<${#stages[@]}; i++)); do
        local current_stage="${stages[$i]}"
        if ! submit_and_monitor "$struct_dir" "$current_stage"; then
            echo "Error in $current_stage calculation for $struct_dir" >> "$error_log_file"
            return 1
        fi
    done
    return 0
}

# Function to process failed calculations
process_failed_calcs() {
    declare -A structure_errors  # Store the earliest failed stage for each structure
    declare -A active_jobs      # Track currently running jobs for each structure
    declare -A current_stage    # Track current calculation stage for each structure
    
    # First pass: identify the earliest failed stage for each structure
    while IFS=' ' read -r struct_dir calc_type; do
        # Only store the earliest failed stage for each structure
        case $calc_type in
            "SC")   structure_errors[$struct_dir]="SC" ;;
            "DIAG") 
                if [ "${structure_errors[$struct_dir]}" != "SC" ]; then
                    structure_errors[$struct_dir]="DIAG"
                fi
                ;;
            "GW0")  
                if [ "${structure_errors[$struct_dir]}" != "SC" ] && [ "${structure_errors[$struct_dir]}" != "DIAG" ]; then
                    structure_errors[$struct_dir]="GW0"
                fi
                ;;
            "BSE")  
                if [ "${structure_errors[$struct_dir]}" != "SC" ] && [ "${structure_errors[$struct_dir]}" != "DIAG" ] && [ "${structure_errors[$struct_dir]}" != "GW0" ]; then
                    structure_errors[$struct_dir]="BSE"
                fi
                ;;
        esac
        current_stage[$struct_dir]=${structure_errors[$struct_dir]}
    done < "$error_log_file"

    # Submit initial jobs for all structures
    for struct_dir in "${!structure_errors[@]}"; do
        echo "Starting calculations for structure $struct_dir from ${structure_errors[$struct_dir]}" >> "$error_log_file"
        
        # Check current number of nodes in use
        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        while [ $sq_results -ge 60 ]; do
            sleep 30
            sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        done

        if ! prepare_directory "${current_stage[$struct_dir]}" "$struct_dir"; then
            echo "Failed to prepare directory for ${current_stage[$struct_dir]} calculation in $struct_dir" >> "$error_log_file"
            continue
        fi

        cd "$struct_dir/Optics/${current_stage[$struct_dir]}" || continue
        slurm_output=$(sbatch sbp.sh)
        if [ $? -eq 0 ]; then
            job_id=$(echo $slurm_output | awk '{print $4}')
            active_jobs[$struct_dir]=$job_id
            echo "Submitted ${current_stage[$struct_dir]} job for $struct_dir with ID $job_id" >> "$error_log_file"
        fi
        cd "$calc_dir" || exit
    done

    # Monitor jobs and submit next stages as needed
    while [ ${#active_jobs[@]} -gt 0 ]; do
        for struct_dir in "${!active_jobs[@]}"; do
            job_id=${active_jobs[$struct_dir]}
            current_type=${current_stage[$struct_dir]}

            if job_completed_successfully "$job_id" "$struct_dir" "$current_type"; then
                unset active_jobs[$struct_dir]
                
                # Determine and submit next stage
                case $current_type in
                    "SC")   next_stage="DIAG" ;;
                    "DIAG") next_stage="GW0" ;;
                    "GW0")  next_stage="BSE" ;;
                    "BSE")  next_stage="" ;;
                esac

                if [ -n "$next_stage" ]; then
                    # Check node usage before submitting next job
                    sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
                    while [ $sq_results -ge 60 ]; do
                        sleep 30
                        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
                    done

                    if prepare_directory "$next_stage" "$struct_dir"; then
                        cd "$struct_dir/Optics/$next_stage" || continue
                        slurm_output=$(sbatch sbp.sh)
                        if [ $? -eq 0 ]; then
                            job_id=$(echo $slurm_output | awk '{print $4}')
                            active_jobs[$struct_dir]=$job_id
                            current_stage[$struct_dir]=$next_stage
                            echo "Submitted $next_stage job for $struct_dir with ID $job_id" >> "$error_log_file"
                        fi
                        cd "$calc_dir" || exit
                    fi
                else
                    echo "Completed all calculations for $struct_dir" >> "$error_log_file"
                fi
            elif ! squeue -u $USER | grep -q "$job_id"; then
                # Job failed
                echo "Job $job_id failed for $struct_dir (${current_stage[$struct_dir]})" >> "$error_log_file"
                unset active_jobs[$struct_dir]
            fi
        done
        sleep 300
    done
}

# Function to check job completion
job_completed_successfully() {
    local job_id=$1
    local struct_dir=$2
    local calc_type=$3
    local timestamp_file="/tmp/last_job_check_${struct_dir}_${calc_type}"

    # Convert to absolute path if it's not already
    if [[ "$struct_dir" != /* ]]; then
        struct_dir="$calc_dir/$struct_dir"
    fi
    
    local slurm_file="$struct_dir/Optics/$calc_type/slurm-${job_id}.out"
    local outcar_file="$struct_dir/Optics/$calc_type/OUTCAR"

    # Check if job is still in queue
    if squeue -u $USER | grep -q "$job_id"; then
        # Only log every 15 minutes (900 seconds)
        current_time=$(date +%s)
        if [ ! -f "$timestamp_file" ] || [ $((current_time - $(cat "$timestamp_file"))) -ge 900 ]; then
            echo "Job $job_id still running for $struct_dir ($calc_type)" >> "$error_log_file"
            echo "$current_time" > "$timestamp_file"
        fi
        return 1  # Job is still running
    fi

    # Add a small delay to ensure slurm output file is written
    sleep 5

    # Clean up timestamp file when job is done
    rm -f "$timestamp_file"

    echo "Checking job completion status from slurm file: $slurm_file" >> "$error_log_file"
    
    # Check if slurm output file exists and is readable
    if [ ! -f "$slurm_file" ]; then
        echo "Slurm output file not found: $slurm_file" >> "$error_log_file"
        return 1
    fi

    if [ ! -r "$slurm_file" ]; then
        echo "Slurm output file not readable: $slurm_file" >> "$error_log_file"
        return 1
    fi

    # Debug: print file contents
    echo "Slurm file contents:" >> "$error_log_file"
    cat "$slurm_file" >> "$error_log_file"
    
    # Check for common SLURM/VASP error patterns
    if grep -q -E "CANCELLED|error|failed|BAD TERMINATION|SIGSEGV|segmentation fault occurred" "$slurm_file"; then
        echo "Error found in job $job_id for $struct_dir ($calc_type)" >> "$error_log_file"
        return 1
    fi
    
    # Check for VASP specific completion markers in OUTCAR
    if [[ -f "$outcar_file" ]] && grep -q "General timing and accounting informations for this job:" "$outcar_file"; then
        case $calc_type in
            "SC")
                if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]]; then
                    echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$error_log_file"
                    return 0
                fi
                ;;
            "DIAG")
                if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                   [[ -s "$struct_dir/Optics/$calc_type/WAVEDER" ]]; then
                    echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$error_log_file"
                    return 0
                fi
                ;;
            "GW0")
                if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                   ls "$struct_dir/Optics/$calc_type/"*.tmp 1> /dev/null 2>&1; then
                    echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$error_log_file"
                    return 0
                fi
                ;;
            "BSE")
                if [[ -s "$struct_dir/Optics/$calc_type/WAVECAR" ]] && \
                   [[ -s "$struct_dir/Optics/$calc_type/WAVEDER" ]]; then
                    echo "Job $job_id completed successfully for $struct_dir ($calc_type)" >> "$error_log_file"
                    return 0
                fi
                ;;
        esac
    fi

    echo "Job $job_id completion check failed for $struct_dir ($calc_type)" >> "$error_log_file"
    return 1
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

# Start resubmission of failed calculations
echo "Starting resubmission of failed calculations..."
process_failed_calcs

echo "All calculations resubmitted and completed"