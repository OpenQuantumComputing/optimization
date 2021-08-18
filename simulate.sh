#!/bin/bash

read -p "Number of routes: " routes
read -p "Sols ('high' or 'low'): " sols

for file in ../data/tailassignment_samples/npy_samples/*.npy; do
    filename="$(basename $file)"

    r=$(echo $filename | cut -d'_' -f 2)
    s=$(echo $filename | cut -d'_' -f 4)
    i=$(echo $filename | cut -d'_' -f 5)

    i=$(echo $i | cut -d'.' -f 1)
    
    if [ "$r" = "$routes" ];
    then
        if ( [ "$sols" = "high" ] && [ "$s" != 1 ] ) || ( [ "$sols" = "low" ] && [ "$s" = 1 ] );
        then
            python run_single_sim.py $filename $i &
        fi
    fi
done
