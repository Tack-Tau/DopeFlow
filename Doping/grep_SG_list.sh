#!/bin/bash
cat sorted_E_form_min_list.dat | awk '{print $1}' | while read -r lines ; do SG="$(aflow --aflowSG < $lines/Relax/CONTCAR)" ; echo $lines $SG ; done > sorted_SG_list.dat
grep -v "#1" sorted_SG_list.dat | awk '{print $1}' > phonon_list
