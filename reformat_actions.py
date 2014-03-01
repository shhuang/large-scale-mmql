#!/usr/bin/env python
# (One-off script)
# Renames the keys of the input actions.h5 file, to fit the requirements of
# the MMQL model in reinforcement-lfd/rope_qlearn.py
# Also deletes actions corresponding to ar_demo.
#
# WARNING: This changes the input actions.h5 file -- you may want to make a
# backup of it

import argparse, h5py

ARMS = {'l':'leftarm', 'r':'rightarm'}

def reformat_keys(fname):
    actions = h5py.File(fname, 'r+')
    for k in actions.keys():
        if k.startswith('ar_demo'):
            del actions[k]
            continue
        for lr in 'lr':
            g = actions[k].create_group('{}_gripper_tool_frame'.format(lr))
            g['hmat'] = actions[k][ARMS[lr]][()]
            del actions[k][ARMS[lr]]
            actions[k]['{}_gripper_joint'.format(lr)] = \
                actions[k]['{}_gripper'.format(lr)][()]
            del actions[k]['{}_gripper'.format(lr)]
    actions.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actionsfile", type=str)
    args = parser.parse_args()
    
    reformat_keys(args.actionsfile)

if __name__ == "__main__":
    main()
