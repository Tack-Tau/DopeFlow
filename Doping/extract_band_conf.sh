#!/bin/bash

# Ensure necessary files are available
HSP_FILE="HIGH_SYMMETRY_POINTS"
POSCAR="POSCAR"  # Replace with the actual path to your POSCAR file
KPOINTS="KPOINTS"  # Replace with the actual path to your KPOINTS file

if [[ ! -f "$HSP_FILE" ]]; then
    echo "HIGH_SYMMETRY_POINTS is missing."
    exit 1
fi

# Read high-symmetry points and labels from HIGH_SYMMETRY_POINTS
high_symmetry_points=()
high_symmetry_labels=()
while read -r line; do
    if [[ "$line" =~ ^([0-9.-]+)\ +([0-9.-]+)\ +([0-9.-]+)\ +(.*) ]]; then
        x=${BASH_REMATCH[1]}
        y=${BASH_REMATCH[2]}
        z=${BASH_REMATCH[3]}
        label=${BASH_REMATCH[4]}
        high_symmetry_points+=("$x $y $z")
        high_symmetry_labels+=("$label")
    fi
done < "$HSP_FILE"

# Read ATOM_NAME from POSCAR
ATOM_NAME=$(sed -n '6s/^\s*//; 6s/\s*$//; 6p' "$POSCAR")

# Read DIM from KPOINTS
DIM=$(sed -n '4p' "$KPOINTS" | awk '{print $1, $2, $3}')

# Generate band.conf automatically
BAND_CONF="band.conf"
cat << EOF > "$BAND_CONF"
ATOM_NAME = $ATOM_NAME
DIM = $DIM
PRIMITIVE_AXES = AUTO
EOF

# Write BAND line
echo -n "BAND = " >> "$BAND_CONF"
for point in "${high_symmetry_points[@]}"; do
    echo -n "$point " >> "$BAND_CONF"
done
echo "" >> "$BAND_CONF"

# Write BAND_LABELS line
echo -n "BAND_LABELS = " >> "$BAND_CONF"
for label in "${high_symmetry_labels[@]}"; do
    if [[ "$label" == \* || \
          "$label" =~ ^\\(Gamma|Delta|Sigma|Pi|Lambda|Omega|\
alpha|beta|theta|phi|psi|Psi|Theta|Xi|\
chi|mu|nu|tau|epsilon|varepsilon|\
kappa|rho|zeta|eta)$ ]]; then
        printf '$%s$ ' "$label" >> "$BAND_CONF"
    else
        printf '%s ' "$label" >> "$BAND_CONF"
    fi
done
echo "" >> "$BAND_CONF"

# Write BAND_POINTS line
echo "BAND_POINTS = 101" >> "$BAND_CONF"

echo "Generated $BAND_CONF successfully."

