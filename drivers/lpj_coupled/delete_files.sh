#!/bin/bash

rm -r temp_lpj
rm temp_output/*
rm debugging/*
rm myjob*
for f in ll_output/*; do rm "$f"/*; done

echo "finished deleting files"
