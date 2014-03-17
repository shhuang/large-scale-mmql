#!/usr/bin/env python

# Reformats the actions holdout set into one that can be the input to the MMQL
# model's evaluation phase.

import argparse, h5py
from rapprentice import rope_initialization

def reformat_holdout(orig_holdout_fname, output_holdout_fname):
    orig_holdout = h5py.File(orig_holdout_fname, 'r')
    output_holdout = h5py.File(output_holdout_fname, 'w')
    i = 0
    for k in orig_holdout:
        if not k.endswith('seg00'): # Only add starting states to holdout set
            continue
        g = output_holdout.create_group(str(i))
        g['demo_id'] = str(k)
        g['rope_nodes'] = orig_holdout[k]['cloud_xyz'][()]
        #g['rope_nodes'] = rope_initialization.find_path_through_point_cloud(
        #        orig_holdout[k]['cloud_xyz'][()],
        #        perturb_peak_dist=None, num_perturb_points=0)
        i += 1
    orig_holdout.close()
    output_holdout.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("holdout_actions_file", type=str)
    parser.add_argument("output_holdout_file", type=str)
    args = parser.parse_args()

    reformat_holdout(args.holdout_actions_file, args.output_holdout_file)

if __name__ == "__main__":
    main()
