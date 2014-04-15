#!/usr/bin/env python
# (One-off script)
# Transforms point clouds in the input actions.h5 file by the matrix 'init_tfm'
# in the provided inittfmfile. This matrix is the transformation from the
# camera frame to the robot frame.

import argparse, h5py

ARMS = {'l':'leftarm', 'r':'rightarm'}

def reformat_keys(fname, outputfile, inittfmfile):
    actions = h5py.File(fname, 'r')
    outputf = h5py.File(outputfile, 'w')
    init_tfm = h5py.File(inittfmfile, 'r')['init_tfm'][()]

    for k in actions.keys():
        outputf.copy(actions[k], k)
        for data in outputf[k].keys():
            if data.endswith('cloud_xyz'): # For 'full_cloud_xyz' & 'cloud_xyz'
                cloud_xyz = outputf[k][data][()]
                cloud_xyz = cloud_xyz.dot(init_tfm[:3,:3].T) + \
                        init_tfm[:3,3][None,:]
                del outputf[k][data]
                outputf[k][data] = cloud_xyz

    actions.close()
    outputf.close()
    init_tfm.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actionsfile", type=str)
    parser.add_argument("outputfile", type=str)
    parser.add_argument("--inittfmfile")
    args = parser.parse_args()
    
    reformat_keys(args.actionsfile, args.outputfile, args.inittfmfile)

if __name__ == "__main__":
    main()
