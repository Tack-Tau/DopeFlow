#!/bin/bash

# Ensure necessary files are available
POSCAR="POSCAR"  # Replace with the actual path to your POSCAR file

# Check if POSCAR file exists
if [[ ! -f "$POSCAR" ]]; then
    echo "POSCAR file is missing."
    exit 1
fi

# Define the output file
output_file="HIGH_SYMMETRY_POINTS"

# Run AFLOW command and capture output
aflow_output=$(aflow --kpath < "$POSCAR")
if [[ $? -ne 0 ]]; then
    echo "AFLOW command failed."
    exit 1
fi

# Initialize variables
inside_kpoints=false
previous_line=""
output="High-symmetry points (in fractional coordinates).\n"

# Process the aflow output line by line
while IFS= read -r line; do
  # Check if we're in the k-points section
  if [[ $line == *"KPOINTS TO RUN"* ]]; then
    inside_kpoints=true
  elif [[ $line == *"END"* ]]; then
    inside_kpoints=false
  fi

  # Process lines within the k-points section
  if $inside_kpoints; then
    # Extract k-point lines that contain coordinates
    if [[ $line =~ ^[[:space:]]*([0-9\.-]+)[[:space:]]+([0-9\.-]+)[[:space:]]+([0-9\.-]+)[[:space:]]+! ]]; then
      # Get the coordinates and label
      coordinates="${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
      label=$(echo "$line" | awk '{print $NF}')

      # Skip consecutive duplicate k-points
      if [[ "$coordinates $label" != "$previous_line" ]]; then
        output+="$coordinates $label\n"
        previous_line="$coordinates $label"
      fi
    fi
  fi
done <<< "$aflow_output"

# Write the output to the file
echo -e "$output" > "$output_file"

# Print the output file path
echo "Converted k-points have been written to $output_file"

