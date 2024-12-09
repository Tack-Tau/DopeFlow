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
        if grep -q -E "JOB|CANCELLED|error|failed" $line/Optics/$calc/slurm-* 2>/dev/null; then
            error_count=$((error_count+1))
            echo "$line $calc" >> "$error_log_file"
        fi
    done < "$dir"
    return $error_count
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

# Function to submit jobs with monitoring
submit_jobs() {
    local line=$1
    local calc_type=$2

    mkdir -p "$line/Optics/$calc_type"
    cd "$line/Optics/$calc_type" || return

    case $calc_type in
        "SC")
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cp ../../Relax/CONTCAR ./POSCAR
            cp $calc_dir/POTCAR_GW ./POTCAR
            cp $calc_dir/INCAR_SC ./INCAR
            cp $calc_dir/sbp_SC.sh ./sbp.sh
            ;;
        "DIAG")
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cp ../../Relax/CONTCAR ./POSCAR
            cp $calc_dir/POTCAR_GW ./POTCAR
            cp ../SC/WAVECAR ./
            cp $calc_dir/INCAR_DIAG ./INCAR
            cp $calc_dir/sbp_DIAG.sh ./sbp.sh
            ;;
        "GW0")
            rm !("POSCAR"|"POTCAR"|"INCAR"|"sbp.sh") 2> /dev/null
            cp ../../Relax/CONTCAR ./POSCAR
            cp $calc_dir/POTCAR_GW ./POTCAR
            cp ../DIAG/WAVECAR ./
            cp ../DIAG/WAVEDER ./
            cp $calc_dir/INCAR_GW0 ./INCAR
            cp $calc_dir/sbp_GW0.sh ./sbp.sh
            ;;
        "BSE")
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

    # Wait for job completion
    while true; do
        if [ -f "slurm-$slurm_job_id.out" ]; then
            if grep -q "COMPLETED" "slurm-$slurm_job_id.out"; then
                return 0
            elif grep -q -E "JOB|CANCELLED|error|failed" "slurm-$slurm_job_id.out"; then
                return 1
            fi
        fi
        sleep 30
    done
}

# Process each error line and resubmit jobs
while IFS=' ' read -r line calc_type; do
    echo "Resubmitting calculations for structure $line starting from $calc_type"
    case $calc_type in
        "SC")   calc_sequence=("SC" "DIAG" "GW0" "BSE") ;;
        "DIAG") calc_sequence=("DIAG" "GW0" "BSE") ;;
        "GW0")  calc_sequence=("GW0" "BSE") ;;
        "BSE")  calc_sequence=("BSE") ;;
    esac

    for calc in "${calc_sequence[@]}"; do
        if ! submit_jobs "$line" "$calc"; then
            echo "Error in $calc calculation for $line"
            break
        fi
    done
done < "$error_log_file"

echo "All optical calculations resubmitted"