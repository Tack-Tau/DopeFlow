#!/bin/bash

# Files to store categorized results
direct_file="Direct_dir"
indirect_file="Indirect_dir"
metallic_file="Metallic_dir"

# Ensure result files are empty before starting
> $direct_file
> $indirect_file
> $metallic_file

# Get the number of structures from the uniq_poscar_list
N_dir=$(cat -n ../aflow_sym/uniq_poscar_list | tail -1 | awk -F ' ' '{print $1}')

# Iterate through structure directories named as "1", "2", ..., "N_dir"
for struct_dir in $(seq 1 $N_dir); do
    struct_path="${struct_dir}/Band/"
    echo "Processing: $struct_path"

    # Ensure the directory exists
    if [[ -d "$struct_path" ]]; then
        # Check if slurm log exists and has errors
        slurm_file=$(find $struct_path -name "slurm-*")
        if [[ -n $slurm_file ]]; then
            if grep -q -i "error" "$slurm_file"; then
                echo "Error detected in $slurm_file. Skipping $struct_path."
                continue
            fi
        fi

        # Perform VASPkit post-processing for band structure
        cd "$struct_path" || continue
        if [[ ! -f "BAND_GAP" ]]; then
            echo "Running VASPkit band structure analysis in $struct_path"
            vaspkit -task 303 > vaspkit.out
        fi

        # Check if BAND_GAP file was generated
        if [[ -f "BAND_GAP" ]]; then
            # Extract the "Band Character" line from the BAND_GAP file
            band_character=$(grep "Band Character:" BAND_GAP | awk '{print $3}')

            # Categorize based on the band character
            case $band_character in
                Direct)
                    echo "$struct_path" >> "../$direct_file"
                    ;;
                Indirect)
                    echo "$struct_path" >> "../$indirect_file"
                    ;;
                Metallic)
                    echo "$struct_path" >> "../$metallic_file"
                    ;;
                Semimetallic)
                    echo "$struct_path" >> "../$metallic_file"
                    ;;
                *)
                    echo "Unknown band character in $struct_path. Check BAND_GAP."
                    ;;
            esac
        else
            echo "BAND_GAP file not found in $struct_path. Skipping."
        fi
        cd - > /dev/null || continue
    else
        echo "Directory $struct_path does not exist. Skipping."
    fi
done

echo "Post-processing complete. Results stored in:"
echo "  - $direct_file"
echo "  - $indirect_file"
echo "  - $metallic_file"

