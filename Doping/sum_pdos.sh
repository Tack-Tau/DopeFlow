#!/bin/bash

input_file="projected_dos.dat"
output_file="total_dos.dat"

# Read the PDOS line from pdos.conf
pdos_line=$(grep "^PDOS" pdos.conf)
pdos_values=${pdos_line#*=}

# Split the PDOS values into groups by comma
IFS=',' read -r -a pdos_groups <<< "$pdos_values"

# Initialize arrays to hold column indices for each element
element_indices=()

# Extract element names from POSCAR and ensure multiple elements are handled correctly
element_names=($(sed -n '6p' POSCAR | tr -s ' '))

# Extract column indices for each element and adjust for 1-based indexing
for group in "${pdos_groups[@]}"; do
    group=$(echo $group | sed 's/^ *//;s/ *$//')
    IFS=' ' read -r -a indices <<< "$group"
    adjusted_indices=()
    for index in "${indices[@]}"; do
        adjusted_indices+=($((index + 1)))
    done
    element_indices+=("${adjusted_indices[*]}")
done

# Create the output file header
header_line="# Frequency_THz"
for element in "${element_names[@]}"; do
    header_line+=" Total_DOS_$element"
done
echo "$header_line" > $output_file

# Generate awk script to process the input file
awk_script='NR>1 { total_dos="";'

# Add code to sum the columns for each element
for i in "${!element_names[@]}"; do
    columns=(${element_indices[$i]})
    sum_code="total_dos_${element_names[$i]} = 0;"
    for col in "${columns[@]}"; do
        sum_code+="total_dos_${element_names[$i]} += \$$col;"
    done
    awk_script+="$sum_code"
    awk_script+=' total_dos = (total_dos ? total_dos FS : "") total_dos_'${element_names[$i]}';'
done

# Complete the awk script
awk_script+=' print $1, total_dos; }'

# Run awk script and append the results to the output file
awk "$awk_script" $input_file >> $output_file

