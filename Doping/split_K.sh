#!/bin/bash

parts_total="$2";
input="$1";

parts=$((parts_total))
for i in $(seq 1 $((parts_total-1))); do
  lines=$(wc -l "$input" | cut -f 1 -d" ")
  #n is rounded, 1.3 to 2, 1.6 to 2, 1 to 1
  n=$(awk  -v lines=$lines -v parts=$parts 'BEGIN {
    n = lines/parts;
    rounded = sprintf("%.0f", n);
    if(n>rounded){
      print rounded + 1;
    }else{
      print rounded;
    }
  }');
  head -$n "$input" > KPATH.in.${i}
  tail -$((lines-n)) "$input" > .tmp${i}
  input=".tmp${i}"
  parts=$((parts-1));
done
mv .tmp$((parts_total-1)) KPATH.in.$((parts_total-0))
rm .tmp*
