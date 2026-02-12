#!/bin/bash

# Function to read values from phonopy.yaml
read_phonopy_yaml() {
    local key=$1
    grep -A 0 "$key" phonopy.yaml | awk -F ":" '{print $2}' | tr -d '"'
}

# Main script

# Read ATOM_NAME from POSCAR
atom_name=$(sed -n '6p' POSCAR | tr -s ' ')

# Read atom counts from POSCAR
atom_counts=$(sed -n '7p' POSCAR)

# Extract elements and their counts
elements=($atom_name)
counts=($atom_counts)

# Generate PDOS string
pdos=""
index=1
for i in "${!elements[@]}"; do
    element_count=${counts[$i]}
    for ((j=0; j<element_count; j++)); do
        pdos+="$index "
        ((index++))
    done
    # Remove the trailing space and add a comma if it's not the last element
    pdos=$(echo $pdos | sed 's/ $//')
    if [ $i -ne $((${#elements[@]} - 1)) ]; then
        pdos+=", "
    fi
done

# Read DIM from phonopy.yaml
dimensions=$(read_phonopy_yaml "dim")

# Double the dimensions to get MP
mp=$(echo $dimensions | awk '{print $1*4, $2*4, $3*4}')

# Generate pdos.conf file
cat << EOF > pdos.conf
ATOM_NAME = $atom_name
DIM = $dimensions
MP = $mp
PDOS = $pdos
EOF

# Run phonopy to plot and save the PDOS
phonopy -p -s pdos.conf

