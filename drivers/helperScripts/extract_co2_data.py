#!/bin/env python3

import itertools

def process_file(file_name_in, file_name_out, first_year, last_year, num_of_repetitions, first_out_year):
    result = []

    with open(file_name_in, "r") as f:
        for line in f:
            items = line.split()
            year = int(items[0])
            value = float(items[1])
            if year >= first_year and year <= last_year:
                result.append(value)

    num_of_elements = len(result)

    print(f"Num of initial values: {num_of_elements}\n")

    repeated = itertools.islice(itertools.cycle(result), num_of_elements * num_of_repetitions)

    with open(file_name_out, "w") as f:
        for count, value in enumerate(repeated, start=first_out_year):
            f.write(f"{count} {value}\n")


if __name__ == "__main__":
    file_name_in = "co2_TraCE_21ka_1990CE.txt"
    file_name_out = "co2_TraCE_21ka_1990CE_repeat.txt"
    first_year = -20000
    last_year = -19800
    num_of_repetitions = 20
    first_out_year = -5000

    process_file(file_name_in, file_name_out, first_year, last_year, num_of_repetitions, first_out_year)
