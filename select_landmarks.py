#!/usr/bin/env python

# Create landmarks file by randomly selecting args.num_landmarks items from the
# input labels file.

import argparse, h5py, random

def select_landmarks(labelfile, landmarkfile, n):
    labels = h5py.File(labelfile, 'r')
    landmarks = h5py.File(landmarkfile, 'w')
    landmark_keys = sorted(random.sample(labels.keys(), n), key=lambda a:int(a))
    landmark_ctr = 0
    for k in landmark_keys:
        g = landmarks.create_group(str(landmark_ctr))
        g['cloud_xyz'] = labels[k]['cloud_xyz'][()]
        g['action'] = labels[k]['action'][()]
        g['knot'] = labels[k]['knot'][()]
        landmark_ctr += 1
    labels.close()
    landmarks.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labelfile", type=str)
    parser.add_argument("landmarkfile", type=str)
    parser.add_argument("--num_landmarks", type=int, default=200)
    args = parser.parse_args()

    select_landmarks(args.labelfile, args.landmarkfile, args.num_landmarks)

if __name__ == "__main__":
    main()
