#!/usr/bin/env python

# Fix 1: Append constraints from second input file onto the first input file
# Fix 2: Reassign names to the slack variables so that they are unique (and
#        not duplicated across distributed machines)

import argparse, h5py

def append_constraints(main_constraints_fname, constraints_to_append_fname):
    main_constraints = h5py.File(main_constraints_fname, 'r+')
    extra_constraints = h5py.File(constraints_to_append_fname, 'r')
    start_index = len(main_constraints)
    for i in range(len(extra_constraints)):
        main_constraints.copy(extra_constraints[str(i)], str(start_index + i))
    main_constraints.close()
    extra_constraints.close()

def make_slack_names_unique(main_constraints_fname):
    constraints = h5py.File(main_constraints_fname)
    slack_indices = dict()
    slack_mappings = []
    prev_slack = ""
    prev_slack_mapping = ""
    for i in range(len(constraints)):
        slack_name = constraints[str(i)]['xi'][()]
        [slack_type, slack_num] = slack_name.split('_')
        if slack_name != prev_slack:
            if prev_slack:
                slack_mappings.append((i, prev_slack, prev_slack_mapping))
            if slack_type not in slack_indices:
                slack_indices[slack_type] = 0
            slack_num = slack_indices[slack_type]
            slack_indices[slack_type] += 1
            prev_slack = slack_name
            prev_slack_mapping = "{}_{}".format(slack_type, slack_num)

        del constraints[str(i)]['xi']
        constraints[str(i)]['xi'] = prev_slack_mapping

    # TODO: Remove debugging statement
    print slack_mappings
    constraints.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_constraints_file", type=str)
    parser.add_argument("--append_constraints", type=str)
    parser.add_argument("--make_slack_names_unique", action='store_true')
    args = parser.parse_args()

    if args.append_constraints:
        append_constraints(args.main_constraints_file, args.append_constraints)

    if args.make_slack_names_unique:
        make_slack_names_unique(args.main_constraints_file)

if __name__ == "__main__":
    main()
