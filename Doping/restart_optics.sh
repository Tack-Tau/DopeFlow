#!/bin/bash
shopt -s extglob

calc_dir="$PWD"
log_file="$calc_dir/restart_optical_jobs.log"
error_log_file="$calc_dir/optics_error.log"

# Function to submit jobs with monitoring
submit_jobs() {
    local calc_type=$1

    if [ -f job_${calc_type}_restart.log ]; then
        job_count="$(echo "$(tail -1 job_${calc_type}_restart.log | awk -F ' ' '{print $2}') + 1" | bc)"
    else
        job_count=1
    fi

    while IFS= read -r line; do
        # Check current number of nodes in use
        sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        
        # Wait if node usage is full
        while [ $sq_results -ge 60 ]; do
            sleep 30
            sq_results=$(squeue -u $USER -o "%D" | awk '{sum+=$1} END {print sum}')
        done

        case $calc_type in
            "SC")
                mkdir -p "$line/Optics/$calc_type"
                cd "$line/Optics/$calc_type" || continue
                rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                cp ../../Relax/CONTCAR ./POSCAR
                cp $calc_dir/POTCAR_GW ./POTCAR
                cp $calc_dir/INCAR_SC ./INCAR
                cp $calc_dir/sbp_SC.sh ./sbp.sh
                ;;
            "DIAG")
                mkdir -p "$line/Optics/$calc_type"
                cd "$line/Optics/$calc_type" || continue
                rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
                cp ../../Relax/CONTCAR ./POSCAR
                cp $calc_dir/POTCAR_GW ./POTCAR
                cp ../SC/WAVECAR ./
                cp $calc_dir/INCAR_DIAG ./INCAR
                cp $calc_dir/sbp_DIAG.sh ./sbp.sh
                ;;
            "GW0")
                mkdir -p "$line/Optics/$calc_type"
                cd "$line/Optics/$calc_type" || continue
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
                cd "$line/Optics/$calc_type" || continue
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

        # Submit job and capture job ID
        slurm_output=$(sbatch sbp.sh)
        slurm_job_id=$(echo $slurm_output | awk '{print $4}')

        # Log submission details
        echo "Restarted ${calc_type} job for structure directory: ${line}" >> "$log_file"
        echo "Submitted batch job ${slurm_job_id}" >> "$log_file"
        echo "----------------------------------------" >> "$log_file"

        # Check job completion and errors
        while true; do
            if [ -f "$line/$calc_type/slurm-$slurm_job_id.out" ]; then
                if grep -q -E "JOB|CANCELLED|error|failed" "$line/$calc_type/slurm-$slurm_job_id.out"; then
                    echo "Error in ${calc_type} job for structure directory: ${line}" >> "$error_log_file"
                    break
                elif grep -q "COMPLETED" "$line/$calc_type/slurm-$slurm_job_id.out"; then
                    break
                fi
            fi
            sleep 30
        done

        cd $calc_dir || exit
    done < "$calc_dir/Direct_dir"
}

# Initialize log file with timestamp
echo "Starting restart of optical calculations at $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

# Sequential submission of jobs
echo "Restarting SC calculations..."
submit_jobs "SC"

echo "Restarting DIAG calculations..."
submit_jobs "DIAG"

echo "Restarting GW0 calculations..."
submit_jobs "GW0"

echo "Restarting BSE calculations..."
submit_jobs "BSE"

echo "All optical calculations restarted at $(date)" >> "$log_file"
echo "========================================" >> "$log_file"

echo "All optical calculations have been restarted" 