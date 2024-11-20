#!/bin/bash

# Function to generate KPOINTS using VASPKIT and extract q1, q2, q3
generate_kpoints_and_extract_q() {
    local k_spacing=$1
    local output=$(echo -e "102\n2\n$k_spacing\n" | vaspkit)

    # Extract q1, q2, q3 from the output using awk
    local q1=$(echo "$output" | awk '/K-Mesh Size:/ {print $3}')
    local q2=$(echo "$output" | awk '/K-Mesh Size:/ {print $4}')
    local q3=$(echo "$output" | awk '/K-Mesh Size:/ {print $5}')

    # Return q1, q2, q3 as space-separated string
    echo "$q1 $q2 $q3"
}

# Main script

# Specify the desired k-spacing value
k_spacing=0.07

# Generate KPOINTS and extract q1, q2, q3
dimensions=$(generate_kpoints_and_extract_q $k_spacing)

# Run phonopy with calculated dimensions
phonopy -d --dim="$dimensions" 1> /dev/null

