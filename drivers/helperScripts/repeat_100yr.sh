#!/bin/bash

# Prefix for climate files
PREFIX="Nahuelbuta_TraCE21ka_"

NC_SCRIPT='// Start index of first year to consider
           *START_INDEX = 0;
           // Number of years to repeat
           *NUM_OF_YEARS = 100;
           *NUM_OF_MONTHS = NUM_OF_YEARS * 12;
           time@calendar = "365_day";
           *new_size = $time.size - START_INDEX;
           defdim("time2", new_size);
           *t[$time2] = 0;
           @all = get_vars_in();
           for (*vi = 0; vi < @all.size(); vi++) {
               @var_nm = @all(vi);
               if (@var_nm.ndims() != 2) continue;
               print(@var_nm, "%s\n");
               t = 0;
               for (*idx = 0; idx < new_size; idx++) {
                   t(idx) = @var_nm(0, START_INDEX + (idx % NUM_OF_MONTHS));
               };
               @var_nm(0, :) = t;
           }'

for f in $PREFIX*.nc; do
    echo $f;
    ncap2 -O -s "$NC_SCRIPT" "$f" tmp.nc;
    ncpdq -a time,land_id tmp.nc ${f/.nc}_first_100a_repeating.nc;
    rm tmp.nc
done
