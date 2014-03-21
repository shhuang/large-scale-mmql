#!/usr/bin/env python

from __future__ import division

import argparse
import eval_util, sim_util

from rapprentice import colorize, task_execution, planning, resampling, \
        rope_initialization, clouds, math_utils as mu
import pdb, time

import trajoptpy, openravepy
from rope_qlearn import select_feature_fn, warp_hmats, registration_cost, \
        registration
from knot_classifier import isKnot as is_knot
import os, os.path, numpy as np, h5py
from numpy import asarray
from numpy.linalg import norm
import atexit
import IPython as ipy

COLLISION_DIST_THRESHOLD = 0.0
DS_SIZE = .025

"""
Transform from lr_gripper_tool_frame to end_effector_transform.
This is so that you can give openrave the data in the frame it is expecting.
Openrave does IK in end_effector_frame which is different from gripper_tool_frame.
"""
TFM_GTF_EE = np.array([[ 0.,  0.,  1.,  0.],
                       [ 0.,  1.,  0.,  0.],
                       [-1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.]])

GRIPPER_ANGLE_THRESHOLD = 10.0 # Open/close threshold, adjusted for human demonstrations

class GlobalVars:
    unique_id = 0
    actions = None
    gripper_weighting = False
    init_tfm = None

def get_ds_cloud(sim_env, action):
    return clouds.downsample(GlobalVars.actions[action]['cloud_xyz'], DS_SIZE)

def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)

def blueprint(msg):
    print colorize.colorize(msg, "blue", bold=True)

def yellowprint(msg):
    print colorize.colorize(msg, "yellow", bold=True)

def warp_hmats_tfm(xyz_src, xyz_targ, hmat_list, src_interest_pts = None):
    f, src_params, g, targ_params, cost = registration_cost(xyz_src, xyz_targ, src_interest_pts)
    f = registration.unscale_tps(f, src_params, targ_params)
    trajs = {}
    xyz_src_warped = np.zeros(xyz_src.shape)
    for k, hmats in hmat_list:
        # First transform hmats from the camera frame into the frame of the robot
        hmats_tfm = np.asarray([GlobalVars.init_tfm.dot(h) for h in hmats])
        trajs[k] = f.transform_hmats(hmats_tfm)
    xyz_src_warped = f.transform_points(xyz_src)
    return [trajs, cost, xyz_src_warped]

def get_old_joint_traj_ik(sim_env, ee_hmats, prev_vals, i_start, i_end):
    old_joint_trajs = {}
    for lr in 'lr':
        old_joint_traj = []
        link_name = "%s_gripper_tool_frame"%lr
        manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
        manip = sim_env.robot.GetManipulator(manip_name)
        ik_type = openravepy.IkParameterizationType.Transform6D

        all_x = []
        x = []
        for (i, pose_matrix) in enumerate(ee_hmats[link_name]):
            rot_pose_matrix = pose_matrix.dot(TFM_GTF_EE)
            sols = manip.FindIKSolutions(openravepy.IkParameterization(rot_pose_matrix, ik_type),
                    openravepy.IkFilterOptions.CheckEnvCollisions)
            all_x.append(i)
            if sols != []:
                x.append(i)
                reference_sol = None
                for sol in reversed(old_joint_traj):
                    if sol != None:
                        reference_sol = sol
                        break
                if reference_sol is None:
                    if prev_vals[lr] is not None:
                        reference_sol = prev_vals[lr]
                    else:
                        reference_sol = sim_util.PR2_L_POSTURES['side'] if lr == 'l' else sim_util.mirror_arm_joints(sim_util.PR2_L_POSTURES['side'])
                
                sols = [sim_util.closer_angs(sol, reference_sol) for sol in sols]
                norm_differences = [norm(np.asarray(reference_sol) - np.asarray(sol), 2) for sol in sols]
                min_index = norm_differences.index(min(norm_differences))

                old_joint_traj.append(sols[min_index])

                #blueprint("Openrave IK succeeds")
            #else:
                #redprint("Openrave IK fails")

        if len(x) == 0:
            if prev_vals[lr] is not None:
                vals = prev_vals[lr]
            else:
                vals = sim_util.PR2_L_POSTURES['side'] if lr == 'l' else sim_util.mirror_arm_joints(sim_util.PR2_L_POSTURES['side'])

            old_joint_traj_interp = np.tile(vals,(i_end+1-i_start, 1))
        else:
            if prev_vals[lr] is not None:
                old_joint_traj_interp = sim_util.lerp(all_x, x, old_joint_traj, first=prev_vals[lr])
            else:
                old_joint_traj_interp = sim_util.lerp(all_x, x, old_joint_traj)
        
        yellowprint("Openrave IK found %i solutions out of %i."%(len(x), len(all_x)))

        init_traj_close = sim_util.close_traj(old_joint_traj_interp.tolist())
        old_joint_trajs[lr] = np.asarray(init_traj_close)
    return old_joint_trajs

def compute_trans_traj(sim_env, new_xyz, seg_info, ignore_infeasibility=True, animate=False, interactive=False):
    sim_util.reset_arms_to_side(sim_env)
    
    redprint("Generating end-effector trajectory")    
    
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    if GlobalVars.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
    else:
        interest_pts = None
    lr2eetraj, _, old_xyz_warped = warp_hmats_tfm(old_xyz, new_xyz, hmat_list, interest_pts)

    handles = []
    if animate:
        handles.append(sim_env.env.plot3(old_xyz,5, (1,0,0)))
        handles.append(sim_env.env.plot3(new_xyz,5, (0,0,1)))
        handles.append(sim_env.env.plot3(old_xyz_warped,5, (0,1,0)))

    miniseg_starts, miniseg_ends, _ = sim_util.split_trajectory_by_gripper(seg_info, thresh = GRIPPER_ANGLE_THRESHOLD)    
    success = True
    feasible = True
    misgrasp = False
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    full_trajs = []
    prev_vals = {lr:None for lr in 'lr'}

    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

        # figure out how we're gonna resample stuff


        # Use inverse kinematics to get trajectory for initializing TrajOpt,
        # since demonstrations library does not contain joint angle data for
        # left and right arms
        ee_hmats = {}
        for lr in 'lr':
            ee_link_name = "%s_gripper_tool_frame"%lr
            # TODO: Change # of timesteps for resampling?
            ee_hmats[ee_link_name] = resampling.interp_hmats(np.arange(i_end+1-i_start), np.arange(i_end+1-i_start), lr2eetraj[lr][i_start:i_end+1])
        lr2oldtraj = get_old_joint_traj_ik(sim_env, ee_hmats, prev_vals, i_start, i_end)

        #lr2oldtraj = {}
        #for lr in 'lr':
        #    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
        #    old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
        #    #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
        #    if sim_util.arm_moved(old_joint_traj):       
        #        lr2oldtraj[lr] = old_joint_traj   

        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = sim_util.unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        bodypart2traj = {}

        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            print "planning trajectory following"
            new_joint_traj, pose_errs = planning.plan_follow_traj(sim_env.robot, manip_name,
                                                       sim_env.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)
            prev_vals[lr] = new_joint_traj[-1]

            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj
            ################################    
            redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, bodypart2traj.keys()))
        full_traj = sim_util.getFullTraj(sim_env, bodypart2traj)
        full_trajs.append(full_traj)

        for lr in 'lr':
            gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start], GRIPPER_ANGLE_THRESHOLD)
            prev_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1], GRIPPER_ANGLE_THRESHOLD) if i_start != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

        if len(full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD):
                redprint("Trajectory not feasible")
                feasible = False
            if feasible or ignore_infeasibility:
                success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive)
            else:
                success = False

        if not success: break

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    return success, feasible, misgrasp, full_trajs

def simulate_demo_traj(sim_env, new_xyz, seg_info, full_trajs, ignore_infeasibility=True, animate=False, interactive=False):
    sim_util.reset_arms_to_side(sim_env)
    
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    handles = []
    if animate:
        handles.append(sim_env.env.plot3(old_xyz,5, (1,0,0)))
        handles.append(sim_env.env.plot3(new_xyz,5, (0,0,1)))

    miniseg_starts, miniseg_ends, _ = sim_util.split_trajectory_by_gripper(seg_info, thresh = GRIPPER_ANGLE_THRESHOLD)    
    success = True
    feasible = True
    misgrasp = False
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends

    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):      
        if i_miniseg >= len(full_trajs): break           

        full_traj = full_trajs[i_miniseg]

        for lr in 'lr':
            gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start], GRIPPER_ANGLE_THRESHOLD)
            prev_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1], GRIPPER_ANGLE_THRESHOLD) if i_start != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

        if len(full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD):
                redprint("Trajectory not feasible")
                feasible = False
            if feasible or ignore_infeasibility:
                success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive)
            else:
                success = False

        if not success: break

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs
    
def q_value_fn(state, action, fn, weights, w0):
    return np.dot(weights, fn(state, action)) + w0

def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args, sim_env):
    if args.random_seed is not None: np.random.seed(args.random_seed)

    GlobalVars.actions = h5py.File(args.actionfile, 'r')
    if args.subparser_name == "eval":
        GlobalVars.gripper_weighting = args.gripper_weighting

def parse_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile', type=str, nargs='?', default='data/misc/actions.h5')
    parser.add_argument('holdoutfile', type=str, nargs='?', default='data/misc/holdout_set.h5')
    parser.add_argument("fakedatafile", type=str) # required since large-scale actions file does not contain robot positions

    parser.add_argument("--animation", type=int, default=0, help="if greater than 1, the viewer tries to load the window and camera properties without idling at the beginning")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes'], default=[])
    parser.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='.win_prop.txt')
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")

    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument("weightfile", type=str)
    parser_eval.add_argument("--exec_rope_params", type=str, default='default')
    parser_eval.add_argument("--lookahead_rope_params", type=str, default='default')
    parser_eval.add_argument("--lookahead_width", type=int, default=1)
    parser_eval.add_argument('--lookahead_depth', type=int, default=0)
    parser_eval.add_argument('--ensemble', action='store_true')
    parser_eval.add_argument('--rbf', action='store_true')
    parser_eval.add_argument('--landmark_features')
    parser_eval.add_argument('--quad_landmark_features', action='store_true')
    parser_eval.add_argument('--only_landmark', action="store_true")
    parser_eval.add_argument("--quad_features", action="store_true")
    parser_eval.add_argument("--sc_features", action="store_true")
    parser_eval.add_argument("--rope_dist_features", action="store_true")
    parser_eval.add_argument("--traj_features", action="store_true")
    parser_eval.add_argument("--gripper_weighting", action="store_true")
    
    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--replay_rope_params", type=str, default=None, help="if not specified, uses the rope_params that is saved in the result file")

    return parser.parse_args()

def get_unique_id(): 
    GlobalVars.unique_id += 1
    return GlobalVars.unique_id - 1

def eval_on_holdout(args, sim_env):
    feature_fn, _, num_features, actions = select_feature_fn(args)
    
    weightfile = h5py.File(args.weightfile, 'r')
    weights = weightfile['weights'][:]
    w0 = weightfile['w0'][()] if 'w0' in weightfile else 0
    weightfile.close()
    assert weights.shape[0] == num_features, "Dimensions of weights and features don't match. Make sure the right feature is being used"
    
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    tasks = eval_util.get_specified_tasks(args.tasks, args.taskfile, args.i_start, args.i_end)
    holdout_items = eval_util.get_holdout_items(holdoutfile, tasks)

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        print "task %s" % i_task
        sim_util.reset_arms_to_side(sim_env)
        redprint("Replace rope")
        rope_xyz = demo_id_rope_nodes["rope_nodes"][:]
        # Transform rope_nodes from the kinect's frame into the frame of the PR2
        if 'frame' not in demo_id_rope_nodes or demo_id_rope_nodes['frame'][()] != 'r':
            redprint("Transforming rope into frame of robot")
            rope_xyz = rope_xyz.dot(GlobalVars.init_tfm[:3,:3].T) + GlobalVars.init_tfm[:3,3][None,:]
        rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz)

        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, args.exec_rope_params)

        for i_step in range(args.num_steps):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env)

            redprint("Observe point cloud")
            new_xyz = sim_env.sim.observe_cloud()
            state = ("eval_%i"%get_unique_id(), new_xyz)
            
            redprint("Choosing an action")
            q_values = [q_value_fn(state, action, feature_fn, weights, w0) for action in actions]
            q_values_root = q_values
            time_machine.set_checkpoint('depth_0_%i'%i_step, sim_env)

            assert args.lookahead_width>= 1, 'Lookahead branches set to zero will fail to select any action'
            agenda = sorted(zip(q_values, actions), key = lambda v: -v[0])[:args.lookahead_width]
            agenda = [(v, a, 'depth_0_%i'%i_step, a) for (v, a) in agenda] # state is (value, most recent action, checkpoint id, root action)
            best_root_action = None
            for depth in range(args.lookahead_depth):
                expansion_results = []
                for (branch, (q, a, chkpt, r_a)) in enumerate(agenda):
                    time_machine.restore_from_checkpoint(chkpt, sim_env, sim_util.get_rope_params(args.lookahead_rope_params))
                    cur_xyz = sim_env.sim.observe_cloud()
                    success, _, _, full_trajs = \
                        compute_trans_traj(sim_env, cur_xyz, GlobalVars.actions[a], animate=args.animation, interactive=False)
                    if args.animation:
                        sim_env.viewer.Step()
                    if is_knot(sim_env.sim.rope.GetControlPoints()):
                        best_root_action = r_a
                        break
                    result_cloud = sim_env.sim.observe_cloud()
                    result_chkpt = 'depth_%i_branch_%i_%i'%(depth+1, branch, i_step)
                    if depth != args.lookahead_depth-1: # don't save checkpoint at the last depth to save computation time
                        time_machine.set_checkpoint(result_chkpt, sim_env)
                    expansion_results.append((result_cloud, a, success, result_chkpt, r_a))
                if best_root_action is not None:
                    redprint('Knot Found, stopping search early')
                    break
                agenda = []
                for (cld, incoming_a, success, chkpt, r_a) in expansion_results:
                    if not success:
                        agenda.append((-np.inf, actions[0], chkpt, r_a)) # TODO why first action?
                        continue
                    next_state = ("eval_%i"%get_unique_id(), cld)
                    q_values = [(q_value_fn(next_state, action, feature_fn, weights, w0), action, chkpt, r_a) for action in actions]
                    agenda.extend(q_values)
                agenda.sort(key = lambda v: -v[0])
                agenda = agenda[:args.lookahead_width]
                first_root_action = agenda[0][-1]
                if all(r_a == first_root_action for (_, _, _, r_a) in agenda):
                    best_root_action = first_root_action
                    redprint('All best actions have same root, stopping search early')
                    break
            if best_root_action is None:
                best_root_action = agenda[0][-1]
            
            time_machine.restore_from_checkpoint('depth_0_%i'%i_step, sim_env, sim_util.get_rope_params(args.exec_rope_params))
            eval_stats = eval_util.EvalStats()
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = \
                compute_trans_traj(sim_env, new_xyz, GlobalVars.actions[best_root_action], animate=args.animation, interactive=args.interactive)
            
            print "BEST ACTION:", best_root_action
            eval_util.save_task_results_step(args.resultfile, sim_env, i_task, i_step, eval_stats, best_root_action, full_trajs, q_values_root)
            
            if is_knot(sim_env.sim.rope.GetControlPoints()):
                redprint("KNOT TIED!")
                break;

        if is_knot(sim_env.sim.rope.GetControlPoints()):
            num_successes += 1
        num_total += 1

        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))

# make args more module (i.e. remove irrelevant args for replay mode)
def replay_on_holdout(args, sim_env):
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    tasks = eval_util.get_specified_tasks(args.tasks, args.taskfile, args.i_start, args.i_end)
    loadresultfile = h5py.File(args.loadresultfile, 'r')
    loadresult_items = eval_util.get_holdout_items(loadresultfile, tasks)

    num_successes = 0
    num_total = 0
    
    for i_task, demo_id_rope_nodes in loadresult_items:
        print "task %s" % i_task
        sim_util.reset_arms_to_side(sim_env)
        redprint("Replace rope")
        rope_nodes, rope_params, _, _ = eval_util.load_task_results_init(args.loadresultfile, i_task)
        # uncomment if the results file don't have the right rope nodes
        #rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        if args.replay_rope_params:
            rope_params = args.replay_rope_params
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, rope_params)

        for i_step in range(len(loadresultfile[i_task]) - (1 if 'init' in loadresultfile[i_task] else 0)):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env)

            redprint("Observe point cloud")
            new_xyz = sim_env.sim.observe_cloud()
    
            eval_stats = eval_util.EvalStats()

            best_action, full_trajs, q_values, trans, rots = eval_util.load_task_results_step(args.loadresultfile, sim_env, i_task, i_step)
            
            time_machine.set_checkpoint('depth_0_%i'%i_step, sim_env)
            time_machine.restore_from_checkpoint('depth_0_%i'%i_step, sim_env, sim_util.get_rope_params(rope_params))
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = simulate_demo_traj(sim_env, new_xyz, GlobalVars.actions[best_action], full_trajs, animate=args.animation, interactive=args.interactive)

            print "BEST ACTION:", best_action

            replay_trans, replay_rots = sim_util.get_rope_transforms(sim_env)
            if np.linalg.norm(trans - replay_trans) > 0 or np.linalg.norm(rots - replay_rots) > 0:
                yellowprint("The rope transforms of the replay rope doesn't match the ones in the original result file by %f and %f" % (np.linalg.norm(trans - replay_trans), np.linalg.norm(rots - replay_rots)))
            else:
                yellowprint("Reproducible results OK")
            
            eval_util.save_task_results_step(args.resultfile, sim_env, i_task, i_step, eval_stats, best_action, full_trajs, q_values)
            
            if is_knot(sim_env.sim.rope.GetControlPoints()):
                break;

        if is_knot(sim_env.sim.rope.GetControlPoints()):
            num_successes += 1
        num_total += 1

        redprint('REPLAY Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def load_simulation(args, sim_env):
    sim_env.env = openravepy.Environment()
    sim_env.env.StopSimulation()
    sim_env.env.Load("robots/pr2-beta-static.zae")
    sim_env.robot = sim_env.env.GetRobots()[0]

    actions = h5py.File(args.actionfile, 'r')
    fakedatafile = h5py.File(args.fakedatafile, 'r')
    GlobalVars.init_tfm = fakedatafile['init_tfm'][()]
    
    init_rope_xyz, _ = sim_util.load_fake_data_segment(sim_env, fakedatafile, args.fake_data_segment, args.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data
    
    # Set table height to correct height of first rope in holdout set
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    first_holdout = holdoutfile[holdoutfile.keys()[0]]
    init_rope_xyz = first_holdout['rope_nodes'][:]
    if 'frame' not in first_holdout or first_holdout['frame'][()] != 'r':
        init_rope_xyz = init_rope_xyz.dot(GlobalVars.init_tfm[:3,:3].T) + GlobalVars.init_tfm[:3,3][None,:]

    table_height = init_rope_xyz[:,2].mean() - .02  # Before: .02
    table_xml = sim_util.make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    sim_env.env.LoadData(table_xml)

    if 'bookshelve' in args.obstacles:
        sim_env.env.Load("data/bookshelves.env.xml")
    if 'boxes' in args.obstacles:
        sim_env.env.LoadData(sim_util.make_box_xml("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12]))
        sim_env.env.LoadData(sim_util.make_box_xml("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08]))

    cc = trajoptpy.GetCollisionChecker(sim_env.env)
    for gripper_link in [link for link in sim_env.robot.GetLinks() if 'gripper' in link.GetName()]:
        cc.ExcludeCollisionPair(gripper_link, sim_env.env.GetKinBody('table').GetLinks()[0])

    sim_util.reset_arms_to_side(sim_env)
    
    if args.animation:
        sim_env.viewer = trajoptpy.GetViewer(sim_env.env)
        if args.animation > 1 and os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                sim_env.viewer.SetWindowProp(*window_prop)
                sim_env.viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            sim_env.viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = sim_env.viewer.GetWindowProp()
                camera_matrix = sim_env.viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."

def main():
    args = parse_input_args()
    setup_log_file(args)

    sim_env = sim_util.SimulationEnv()
    set_global_vars(args, sim_env)
    trajoptpy.SetInteractive(args.interactive)
    load_simulation(args, sim_env)

    if args.subparser_name == "eval":
        eval_on_holdout(args, sim_env)
    elif args.subparser_name == "replay":
        replay_on_holdout(args, sim_env)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
