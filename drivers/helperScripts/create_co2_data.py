#!/usr/bin/python3

import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: create_co2_data.py start end value")
        print("start: start year (for ex. -20049)")
        print("end: end year (for ex. 2014)")
        print("value: the CO2 value that is constant for the whole time period")
        sys.exit(1)
    
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    value = float(sys.argv[3])

    if start >= end:
        print("start must be smaller than end: {} >= {}".format(start, end))
        sys.exit(1)

    with open("co2_data.txt", "wt") as f:
        for year in range(start, end + 1):
            f.write("{} {}\n".format(year, value))
