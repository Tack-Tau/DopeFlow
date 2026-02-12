#!/bin/bash
# Better approach using phonopy-load

for dis in $(seq 1 10); do
    a=$(echo $dis | awk '{print $1*0.01}')

    cat > modulation_${dis}.conf << EOF
MODULATION = 3 3 1, 0.333333 0.333333 0.0 1 $a
EOF

    # Use phonopy-load with existing phonopy_params.yaml
    phonopy-load phonopy_params.yaml --config modulation_${dis}.conf

    if [ -f MPOSCAR ]; then
        cp MPOSCAR POSCAR-${dis}
        echo "Created POSCAR-${dis} with amplitude $a"
    else
        echo "ERROR: MPOSCAR not created for amplitude $a"
        ls -la M* 2>/dev/null
    fi
done
