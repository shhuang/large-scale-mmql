#!/usr/bin/env python
# (One-off script)
# Renames the keys of the input actions.h5 file, to fit the requirements of
# the MMQL model in reinforcement-lfd/rope_qlearn.py.
#
# Also deletes actions corresponding to ar_demo, and within the input start and
# end indices (inclusive), if those are specified.

import argparse, h5py

ARMS = {'l':'leftarm', 'r':'rightarm'}

def reformat_keys(fname, outputfile, ignorei_start, ignorei_end):
    actions = h5py.File(fname, 'r')
    outputf = h5py.File(outputfile, 'w')

    for k in actions.keys():
        if k.startswith('ar_demo'):
            continue
        demo_num = int(k[4:9])
        if demo_num >= ignorei_start and demo_num <= ignorei_end:
            continue

        for seg in actions[k].keys():
            new_seg_k = str(k) + '-' + str(seg)
            seg_g = outputf.create_group(new_seg_k)
            for seg_data in actions[k][seg].keys():
                if seg_data == 'l' or seg_data == 'r':
                    continue
                seg_g[seg_data] = actions[k][seg][seg_data][()]

            for lr in 'lr':
                g = seg_g.create_group('{}_gripper_tool_frame'.format(lr))
                g['hmat'] = actions[k][seg][lr]['tfms'][()]
                seg_g['{}_gripper_joint'.format(lr)] = \
                        actions[k][seg][lr]['pot_angles'][()]

    actions.close()
    outputf.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actionsfile", type=str)
    parser.add_argument("outputfile", type=str)
    parser.add_argument("--ignorei_start", type=int)
    parser.add_argument("--ignorei_end", type=int)
    args = parser.parse_args()
    
    reformat_keys(args.actionsfile, args.outputfile,
            args.ignorei_start, args.ignorei_end)

if __name__ == "__main__":
    main()
