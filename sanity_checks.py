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

def get_avg_point_cloud_size(fname):
    # Returns average number of points in all ['cloud_xyz'] point clouds
    actions = h5py.File(fname, 'r')
    total_points = 0
    for k in actions:
        if 'cloud_xyz' in actions[k]:
            total_points += actions[k]['cloud_xyz'][()].shape[0]
        elif 'rope_nodes' in actions[k]:
            total_points += actions[k]['rope_nodes'][()].shape[0]
    num_actions = len(actions)
    actions.close()
    return float(total_points) / num_actions

def get_avg_traj_length(fname):
    # Returns average number of time steps in trajectory
    actions = h5py.File(fname, 'r')
    total_time_steps = 0
    for k in actions:
        total_time_steps += actions[k]['l_gripper_tool_frame']['hmat'].shape[0]
        total_time_steps += actions[k]['r_gripper_tool_frame']['hmat'].shape[0]
    num_actions = len(actions)
    actions.close()
    return float(total_time_steps) / (2 * num_actions)

def check_concurrent_slack(fname):
    # Prints out index and start of new slack variable
    constraints = h5py.File(fname, 'r')
    if '0' not in constraints:
        print 'ERROR: Must pass in constraints file'
        return
    prev_slack = ""
    for i in range(len(constraints)):
        curr_slack = constraints[str(i)]['xi'][()]
        if curr_slack != prev_slack:
            print "({}, {})".format(i, curr_slack)
            prev_slack = curr_slack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_unique_starting_states', type=str)
    parser.add_argument('--get_point_cloud_size', type=str)
    parser.add_argument('--get_traj_length', type=str)
    parser.add_argument('--check_concurrent_slack', type=str)
    args = parser.parse_args()

    if args.check_unique_starting_states:
        print "Checking that starting states are all unique..."
        if check_unique_starting_states(args.check_unique_starting_states):
            print "    Starting states are all unique"
        else:
            print "    ERROR: Starting states are not all unique"

    if args.get_point_cloud_size:
        print "Calculating average point cloud size..."
        point_cloud_size = get_avg_point_cloud_size(args.get_point_cloud_size)
        print "    Average point cloud size: {} points".format(point_cloud_size)

    if args.get_traj_length:
        print "Calculating average trajectory length..."
        avg_traj_length = get_avg_traj_length(args.get_traj_length)
        print "    Average traj length: {} time steps".format(avg_traj_length)

    if args.check_concurrent_slack:
        print "Printing slack variables to check for concurrency..."
        check_concurrent_slack(args.check_concurrent_slack)

if __name__ == "__main__":
    main()
