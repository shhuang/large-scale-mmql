#!/usr/bin/env python

import argparse, h5py

def check_unique_starting_states(fname):
    # Returns true if all starting states in the file are unique, else false
    actions = h5py.File(fname, 'r')
    starting_states = [k for k in actions.keys() if k.endswith('-seg00')]
    print "    Number of starting states: {}".format(len(starting_states))
    clouds = set()
    for k in starting_states:
        clouds.add(tuple(actions[k]['cloud_xyz'][()].reshape(-1,).tolist()))
    return len(clouds) == len(starting_states)
    actions.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_unique_starting_states', type=str)
    args = parser.parse_args()

    if args.check_unique_starting_states:
        print "Checking that starting states are all unique..."
        if check_unique_starting_states(args.check_unique_starting_states):
            print "    Starting states are all unique"
        else:
            print "    ERROR: Starting states are not all unique"

if __name__ == "__main__":
    main()
