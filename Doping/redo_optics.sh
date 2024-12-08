#!/bin/bash
shopt -s extglob

calc_dir="$PWD"
log_file="$calc_dir/optical_jobs.log"

# Function to check for SLURM errors in completed jobs
check_slurm_errors() {
    local dir=$1
    local calc=$2
    error_count=0
    while IFS= read -r line; do
        if grep -q "JOB\|CANCELLED\|ERROR" "$line/$calc/slurm-*" 2>/dev/null; then
            error_count=$((error_count+1))
        fi
    done < "$dir"
    return $error_count
}

# Function to submit jobs with monitoring
submit_jobs() {
    local calc_type=$1

    if [ -f job_${calc_type}.log ]; then
        job_count="$(echo "$(tail -1 job_${calc_type}.log | awk -F ' ' '{print $2}') + 1" | bc)"
    else
        job_count=1
    fi

    while true; do
        while IFS= read -r line; do
            sq_results=$(squeue -u $USER | sed 1d | wc -l)
            if [ $sq_results -lt 60 ]; then
                case $calc_type in
                    "SC")
                        mkdir -p "$line/Optics/$calc_type"
                        cd "$line/Optics/$calc_type"
                        rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                        cp ../../Relax/CONTCAR ./POSCAR
                        cp $calc_dir/POTCAR_GW ./POTCAR
                        cp $calc_dir/INCAR_SC ./INCAR
                        cp $calc_dir/sbp_SC.sh ./sbp.sh
                        ;;
                    "DIAG")
                        mkdir -p "$line/Optics/$calc_type"
                        cd "$line/Optics/$calc_type"
                        rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                        cp ../../Relax/CONTCAR ./POSCAR
                        cp $calc_dir/POTCAR_GW ./POTCAR
                        cp ../../SC/WAVECAR ./
                        cp $calc_dir/INCAR_DIAG ./INCAR
                        cp $calc_dir/sbp_DIAG.sh ./sbp.sh
                        ;;
                    "GW0")
                        mkdir -p "$line/Optics/$calc_type"
                        cd "$line/Optics/$calc_type"
                        rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                        cp ../../Relax/CONTCAR ./POSCAR
                        cp $calc_dir/POTCAR_GW ./POTCAR
                        cp ../DIAG/WAVECAR ./
                        cp ../DIAG/WAVEDER ./
                        cp $calc_dir/INCAR_GW0 ./INCAR
                        cp $calc_dir/sbp_GW0.sh ./sbp.sh
                        ;;
                    "BSE")
                        mkdir -p "$line/Optics/$calc_type"
                        cd "$line/Optics/$calc_type"
                        rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                        cp ../../Relax/CONTCAR ./POSCAR
                        cp $calc_dir/POTCAR_GW ./POTCAR
                        cp ../GW0/*.tmp ./
                        cp ../GW0/WAVECAR ./
                        cp ../DIAG/WAVEDER ./
                        cp $calc_dir/INCAR_BSE ./INCAR
                        cp $calc_dir/sbp_BSE.sh ./sbp.sh
                        ;;
                esac
                # Capture the sbatch output to get the job ID
                slurm_output=$(sbatch sbp.sh)
                slurm_job_id=$(echo $slurm_output | awk '{print $4}')

                # Log the submission details
                echo "Submitted ${calc_type} job for structure directory: ${line}" >> "$log_file"
                echo "Submitted batch job ${slurm_job_id}" >> "$log_file"
                echo "----------------------------------------" >> "$log_file"

                echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                job_count=$((job_count+1))
                cd $calc_dir
            else
                sleep 10
            fi
        done < "$calc_dir/Direct_dir"

        # Wait for all jobs to complete
        while [ $(squeue -u $USER | sed 1d | wc -l) -gt 0 ]; do
            sleep 30
        done

        # Check for errors
        check_slurm_errors "$calc_dir/Direct_dir" "$calc_type"
        if [ $? -eq 0 ]; then
            break
        fi
        # If errors found, loop will continue and resubmit failed jobs
    done
}

# Add timestamp to log file
echo "Starting optical calculations at $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# Sequential submission of jobs
echo "Starting SC calculations..."
submit_jobs "SC"

echo "Starting DIAG calculations..."
submit_jobs "DIAG"

echo "Starting GW0 calculations..."
submit_jobs "GW0"

echo "Starting BSE calculations..."
submit_jobs "BSE"

echo "All optical calculations completed at $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

echo "All optical calculations completed"
