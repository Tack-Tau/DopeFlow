#!/bin/bash

# Enable extended globbing
shopt -s extglob

# Define verbosity level (0 = minimal, 1 = detailed)
VERBOSE=0

# Define the calculation directory
calc_dir="$PWD"

# Function to print messages based on verbosity level
log() {
    local level=$1
    shift
    if [[ $VERBOSE -ge $level ]]; then
        echo "$@"
    fi
}

# Function to find and collect phonon directory numbers
find_phonon_dir() {
    local phon_dir=""
    for file in POSCAR-*; do
        if [[ -f "$file" ]]; then
            num=$(echo "$file" | grep -oE '[0-9]+')  # Extract the padded number from the filename
            if [[ ! -z "$num" ]]; then
                phon_dir+="$num "  # Append the number to phon_dir
            fi
        fi
    done
    echo "$phon_dir"  # Output the collected numbers as a space-separated string
}

# Function to perform pre-checks for a structure directory
precheck_phonon_dir() {
    local struct_dir="$1"
    local phonon_dir="$calc_dir/$struct_dir/PHON"
    cd "$phonon_dir" || return 1

    # Find the maximum phonon directory number
    phonon_dir_max=$(ls | grep POSCAR- | awk -F '-' '{print $2}' | sort | tail -1)
    if [[ -z "$phonon_dir_max" ]]; then
        log 0 "Error: No phonon directories found for $struct_dir."
        return 1
    fi

    # Ensure all phonon directories up to max exist
    for num in $(find_phonon_dir); do
        if [[ ! -d "$num" ]]; then
            log 0 "Error: Phonon directory $num is missing in $struct_dir."
            return 1
        fi
    done

    # Check for errors in slurm output files
    error_dir=$(for phon_dir in $(find_phonon_dir); do
        error_phon=$(grep -l "error" "$phon_dir"/slurm-* 2>/dev/null | awk -F '/' '{print $1}')
        if [[ -n "$error_phon" ]]; then
            echo "$error_phon"
        fi
    done)
    if [[ -n "$error_dir" ]]; then
        log 0 "Error: Slurm errors detected in the following phonon directories for $struct_dir: $error_dir"
        return 1
    fi

    # Ensure non-empty vasprun.xml exists in all phonon directories
    for num in $(find_phonon_dir); do
        if [[ ! -s "$num/vasprun.xml" ]]; then
            log 0 "Error: Missing or empty vasprun.xml in phonon directory $num for $struct_dir."
            return 1
        fi
    done

    # If all checks passed
    log 1 "Pre-check passed for $struct_dir."
    return 0
}

# Function to format Greek symbols in BAND_LABELS
format_band_labels() {
    sed -i 's/\\Gamma/\$\Gamma\$/g; s/\\Delta/\$\Delta\$/g; s/\\Sigma/\$\Sigma\$/g; s/\\Pi/\$\Pi\$/g' band.conf
}

# Process each structure directory in phonon_list
for struct_dir in $(cat phonon_list); do
    log 0 "Starting post-processing for structure directory: $struct_dir"
    phonon_dir="$calc_dir/$struct_dir/PHON"

    # Ensure phonon directory exists
    if [[ ! -d "$phonon_dir" ]]; then
        log 0 "Error: Phonon directory does not exist for $struct_dir. Skipping."
        continue
    fi

    # Run pre-checks
    if ! precheck_phonon_dir "$struct_dir"; then
        log 0 "Skipping $struct_dir due to pre-check failures."
        continue
    fi

    cd "$phonon_dir" || exit

    # Get the maximum phonon directory number
    phonon_dir_max=$(ls | grep POSCAR- | awk -F '-' '{print $2}' | sort | tail -1)

    # Run phonopy to generate FORCE_SETS
    log 1 "Generating FORCE_SETS..."
    phonopy -f $(find_phonon_dir | xargs -n1 | awk '{print $1 "/vasprun.xml"}') 1> /dev/null && \

    # Run the conversion and band extraction scripts
    log 1 "Running convert_kpath.sh and extract_band_conf.sh..."
    bash convert_kpath.sh 1> /dev/null && bash extract_band_conf.sh 1> /dev/null && \

    # Format BAND_LABELS with LaTeX for Greek symbols
    log 1 "Formatting BAND_LABELS for LaTeX..."
    format_band_labels && \

    # Generate phonon band plot and YAML file
    log 1 "Generating phonon band plot..."
    phonopy -ps band.conf 1> /dev/null && \

    # Create raw data file for the phonon band
    log 1 "Creating raw data file for phonon band..."
    phonopy-bandplot --gnuplot band.yaml > phonon_band.dat && \

    # Run the preprocessing script for high-symmetry points
    log 1 "Preprocessing high-symmetry points..."
    bash preprocess_high_symmetry_points.sh 1> /dev/null && \

    log 0 "Post-processing complete for $struct_dir." || \
    log 0 "Error occurred during post-processing for $struct_dir. Skipping remaining steps."
done

log 0 "All structure directories in phonon_list have been processed."

