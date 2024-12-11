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
    declare -A job_ids  # Associate array to store job IDs for each structure
    declare -A calc_stage  # Track which calculation stage each structure is at
    local status_timestamp_file="/tmp/last_status_print"
    local status_interval=900  # 15 minutes in seconds

    # Create associative arrays to track which structures need which calculations
    declare -A need_sc need_diag need_gw0 need_bse

    # First pass: identify which calculations are needed for each structure
    while IFS=' ' read -r struct_dir calc_type; do
        case $calc_type in
            "SC")
                need_sc[$struct_dir]=1
                need_diag[$struct_dir]=1
                need_gw0[$struct_dir]=1
                need_bse[$struct_dir]=1
                ;;
            "DIAG")
                need_diag[$struct_dir]=1
                need_gw0[$struct_dir]=1
                need_bse[$struct_dir]=1
                ;;
            "GW0")
                need_gw0[$struct_dir]=1
                need_bse[$struct_dir]=1
                ;;
            "BSE")
                need_bse[$struct_dir]=1
                ;;
        esac
    done < "$error_log_file"

    # Process structures that need SC calculations first
    for struct_dir in "${!need_sc[@]}"; do
        echo "Starting calculations from SC for structure $struct_dir" >> "$error_log_file"
        if submit_and_monitor "$struct_dir" "SC"; then
            job_ids[$struct_dir]=$job_id
            calc_stage[$struct_dir]="DIAG"
        fi
    done

    # Process structures that need DIAG calculations (but not SC)
    for struct_dir in "${!need_diag[@]}"; do
        if [[ -z "${need_sc[$struct_dir]}" ]]; then
            echo "Starting calculations from DIAG for structure $struct_dir" >> "$error_log_file"
            if submit_and_monitor "$struct_dir" "DIAG"; then
                job_ids[$struct_dir]=$job_id
                calc_stage[$struct_dir]="GW0"
            fi
        fi
    done

    # Process structures that need GW0 calculations (but not SC or DIAG)
    for struct_dir in "${!need_gw0[@]}"; do
        if [[ -z "${need_sc[$struct_dir]}" && -z "${need_diag[$struct_dir]}" ]]; then
            echo "Starting calculations from GW0 for structure $struct_dir" >> "$error_log_file"
            if submit_and_monitor "$struct_dir" "GW0"; then
                job_ids[$struct_dir]=$job_id
                calc_stage[$struct_dir]="BSE"
            fi
        fi
    done

    # Process structures that need only BSE calculations
    for struct_dir in "${!need_bse[@]}"; do
        if [[ -z "${need_sc[$struct_dir]}" && -z "${need_diag[$struct_dir]}" && -z "${need_gw0[$struct_dir]}" ]]; then
            echo "Starting calculations from BSE for structure $struct_dir" >> "$error_log_file"
            if submit_and_monitor "$struct_dir" "BSE"; then
                job_ids[$struct_dir]=$job_id
                calc_stage[$struct_dir]="BSE"
            fi
        fi
    done

    # Add status printing
    current_time=$(date +%s)
    if [ ! -f "$status_timestamp_file" ] || [ $((current_time - $(cat "$status_timestamp_file"))) -ge $status_interval ]; then
        echo "Current status at $(date):" >> "$error_log_file"
        for struct_dir in "${!calc_stage[@]}"; do
            echo "$struct_dir: ${calc_stage[$struct_dir]} (Job ID: ${job_ids[$struct_dir]})" >> "$error_log_file"
        done
        echo "$current_time" > "$status_timestamp_file"
    fi
}

# Add this new function for job completion checking
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
            echo "Job $job_id still running for $struct_dir ($calc_type)" >> "$error_log_file"
            echo "$current_time" > "$timestamp_file"
        fi
        return 1  # Job is still running
    fi

    # Clean up timestamp file when job is done
    rm -f "$timestamp_file"

    # Check OUTCAR & slurm output file for completion
    local slurm_file="$struct_dir/Optics/$calc_type/slurm-${job_id}.out"
    local outcar_file="$struct_dir/Optics/$calc_type/OUTCAR"
    echo "Checking job completion status from slurm file: $slurm_file" >> "$error_log_file"
    
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
    else
        echo "Slurm output file not found: $slurm_file" >> "$error_log_file"
    fi

    echo "Job $job_id completion check failed for $struct_dir ($calc_type)" >> "$error_log_file"
    return 1  # Default to failure if we can't confirm success
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