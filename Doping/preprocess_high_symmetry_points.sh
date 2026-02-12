#!/bin/bash

# Check if HIGH_SYMMETRY_POINTS file exists
if [ ! -f HIGH_SYMMETRY_POINTS ]; then
  echo "HIGH_SYMMETRY_POINTS file not found!"
  exit 1
fi

# Initialize variables
previous_coords=(0.0000 0.0000 0.0000)
accumulated_norm=0.0000

# Extract the K-point labels and positions, calculate the accumulated L2 norm
awk '/^[[:space:]]*[-0-9.]+[[:space:]]+[-0-9.]+[[:space:]]+[-0-9.]+[[:space:]]+[A-Za-z0-9_\\]+[[:space:]]*$/ {
    print $1, $2, $3, $4
}' HIGH_SYMMETRY_POINTS | while read -r x y z label; do
  # Calculate the L2 norm between consecutive k-points
  dx=$(echo "$x - ${previous_coords[0]}" | bc)
  dy=$(echo "$y - ${previous_coords[1]}" | bc)
  dz=$(echo "$z - ${previous_coords[2]}" | bc)
  norm=$(echo "sqrt($dx * $dx + $dy * $dy + $dz * $dz)" | bc -l)

  # Accumulate the norm
  accumulated_norm=$(echo "$accumulated_norm + $norm" | bc -l)

  # Store the k-point label with its accumulated norm
  printf "%s %.8f\n" "$label" "$accumulated_norm"

  # Update previous coordinates
  previous_coords=($x $y $z)
done > kpoints.txt

echo "Preprocessing completed successfully."

