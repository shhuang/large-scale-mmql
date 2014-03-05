#!/usr/bin/env python

# Append constraints from second input file onto the first input file

import argparse, h5py

def append_constraints(main_constraints_fname, constraints_to_append_fname):
    main_constraints = h5py.File(main_constraints_fname, 'r+')
    extra_constraints = h5py.File(constraints_to_append_fname, 'r')
    start_index = len(main_constraints)
    for i in range(len(extra_constraints)):
        main_constraints.copy(extra_constraints[str(i)], str(start_index + i))
    main_constraints.close()
    extra_constraints.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_constraints_file", type=str)
    parser.add_argument("constraints_to_append", type=str)
    args = parser.parse_args()

    append_constraints(args.main_constraints_file, args.constraints_to_append)

if __name__ == "__main__":
    main()
