#!/usr/bin/env python

# Randomly selects HOLDOUT_SIZE demonstrations for the holdout set, and puts
# the rest in the training set file.

import argparse, h5py, random

HOLDOUT_SIZE = 100
JOINTS = {'l': 'l_gripper_joint', 'r': 'r_gripper_joint'}
FRAMES = {'l': 'l_gripper_tool_frame', 'r': 'r_gripper_tool_frame'}

def select_holdout_keys(fname):
    actions = h5py.File(fname, 'r')
    demo_names = set([k.split('-')[0] for k in actions.keys()])
    print "Number of demos: {}".format(len(demo_names))
    return random.sample(demo_names, HOLDOUT_SIZE)

def create_output_files(actionfile, trainingfile, holdoutfile, holdout_keys):
    actions = h5py.File(actionfile, 'r')
    training = h5py.File(trainingfile, 'w')
    holdout = h5py.File(holdoutfile, 'w')
    for k in actions.keys():
        if k.split('-')[0] in holdout_keys:
            a = holdout.create_group(k)
        else:
            a = training.create_group(k)
        a['cloud_xyz'] = actions[k]['cloud_xyz'][()]
        for lr in 'lr':
            a[JOINTS[lr]] = actions[k][JOINTS[lr]][()]
            a.create_group(FRAMES[lr])
            a[FRAMES[lr]]['hmat'] = actions[k][FRAMES[lr]]['hmat'][()]

    actions.close()
    training.close()
    holdout.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actionfile", type=str)
    parser.add_argument("trainingfile", type=str)
    parser.add_argument("holdoutfile", type=str)
    args = parser.parse_args()

    holdout_keys = select_holdout_keys(args.actionfile)
    create_output_files(args.actionfile, args.trainingfile, args.holdoutfile,
            holdout_keys)

if __name__ == "__main__":
    main()
