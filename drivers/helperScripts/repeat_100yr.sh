#!/bin/bash

PREFIX="LaCampana_TraCE21ka"

NC_SCRIPT='time@calendar="365_day";
           t[time]=0;
           @all=get_vars_in();
           for (vi = 0; vi < @all.size(); vi++) {
               @var_nm = @all(vi);
               if (@var_nm.ndims() != 2) continue;
               print(@var_nm, "%s\n");
               t = 0;
               for (idx = 0; idx < $time.size; idx++) {
                   t(idx) = @var_nm(0, idx % (100*12));
               };
               @var_nm(0,:) = t;
           }'

for f in $PREFIX_*.nc; do
    echo $f;
    ncap2 -O -s $NC_SCRIPT "$f" tmp.nc;
    ncpdq -a time,land_id tmp.nc ${f/.nc}_first_100a_repeating.nc;
done
