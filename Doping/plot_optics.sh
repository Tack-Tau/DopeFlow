#!/bin/bash
input="./Direct_dir" ; while IFS= read -r line ; do cp extract_optics.sh $line/Optics/BSE ; done < "$input"
input="./Direct_dir" ; while IFS= read -r line ; do cd $line/Optics/BSE ; ./extract_optics.sh ; cd ../../.. ; done < "$input"
