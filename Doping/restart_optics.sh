#!/bin/bash
shopt -s extglob

calc_dir="$PWD"
log_file="$calc_dir/restart_optical_jobs.log"

# Function to submit jobs with monitoring
submit_jobs() {
    local calc_type=$1

    if [ -f job_${calc_type}_restart.log ]; then
        job_count="$(echo "$(tail -1 job_${calc_type}_restart.log | awk -F ' ' '{print $2}') + 1" | bc)"
    else
        job_count=1
    fi

    while IFS= read -r line; do
        # Check current number of jobs in queue
        sq_results=$(squeue -u $USER | sed 1d | wc -l)
        
        # Wait if queue is full
        while [ $sq_results -ge 60 ]; do
            sleep 30
            sq_results=$(squeue -u $USER | sed 1d | wc -l)
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

        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}_restart.log
        job_count=$((job_count+1))
        cd $calc_dir || exit
    done < "$calc_dir/Direct_dir"

    # Wait for all jobs of this type to complete
    echo "Waiting for all ${calc_type} jobs to complete..."
    while true; do
        active_jobs=$(squeue -u $USER | grep "${calc_type}" | wc -l)
        if [ $active_jobs -eq 0 ]; then
            echo "All ${calc_type} jobs completed."
            break
        fi
        sleep 30
    done
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