#!/bin/bash

# Prefix for climate files
PREFIX="Nahuelbuta_TraCE21ka_"

NC_SCRIPT='// Start index of first year to consider
           *START_INDEX = 0;
           // Number of years to repeat
           *NUM_OF_YEARS = 100;
           // How often should it be repeated ?
           *NUM_OF_REPETITIONS = 220;

           time@calendar = "365_day";
           *NUM_OF_MONTHS = NUM_OF_YEARS * 12;
           *new_size = NUM_OF_MONTHS * NUM_OF_REPETITIONS;
           defdim("time2", new_size);
           *t[$time2] = 0;
           @all = get_vars_in();

           for (*var_index = 0; var_index < @all.size(); var_index++) {
               @var_name = @all(var_index);
               if (@var_name.ndims() != 2) continue;
               print(@var_name, "%s\n");

               for (*dst_index = 0; dst_index < new_size; dst_index += NUM_OF_MONTHS) {
                   for (*src_index = 0; src_index < NUM_OF_MONTHS; src_index++) {
                       t(dst_index + src_index) = @var_name(0, START_INDEX + src_index);
                   }
               }

               @var_name(0, :) = t;
           }'

for f in $PREFIX*.nc; do
    echo $f;
    ncap2 -O -s "$NC_SCRIPT" "$f" tmp.nc;
    ncpdq -a time,land_id tmp.nc ${f/.nc}_first_100a_repeating.nc;
    rm tmp.nc
done
