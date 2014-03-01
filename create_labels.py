#!/usr/bin/env python

# Generates labelled data (to train MMQL) based on data in the input actions
# file.

import argparse, h5py

def leave_one_out_labelling(actionfile, labelfile):
    #TODO: Implement LOOL
    print "ERROR: Function has not been implemented yet"
    return 0

def same_labelling(actionfile, labelfile):
    # Labelled data matches each starting state in the input actions file to
    # its corresponding action
    actions = h5py.File(actionfile, 'r')
    labels = h5py.File(labelfile, 'w')
    label_ctr = 0
    action_keys = sorted(actions.keys())
    for k in action_keys:
        g = labels.create_group(str(label_ctr))
        if k.endswith('seg00'):
            g['pred'] = str(label_ctr)
        else:
            g['pred'] = str(label_ctr - 1)
        g['cloud_xyz'] = actions[k]['cloud_xyz'][()]
        g['action'] = str(k)
        g['knot'] = 0
        label_ctr += 1
    actions.close()
    labels.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actionfile", type=str)
    parser.add_argument("labelfile", type=str)
    parser.add_argument("labeltype", choices=['same', 'lool'])
    args = parser.parse_args()

    if args.labeltype == "same":
        label_fn = same_labelling
    else:
        label_fn = leave_one_out_labelling
    label_fn(args.actionfile, args.labelfile)

if __name__ == "__main__":
    main()
