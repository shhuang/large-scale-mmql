#!/usr/bin/env python

from __future__ import division

import argparse
import eval_util, sim_util

from rapprentice import colorize, task_execution, planning, resampling, \
        rope_initialization, clouds, plotting_openrave, math_utils as mu
import pdb, time

import trajoptpy, openravepy
from rope_qlearn import select_feature_fn, warp_hmats, registration_cost, \
        registration, get_closing_pts
from knot_classifier import isKnot as is_knot
import ropesim_floating
import os, os.path, numpy as np, h5py
from numpy import asarray
from numpy.linalg import norm
import atexit
import IPython as ipy

"""
example command:
    ./do_task_eval.py --resultfile test_results.h5 --animation 1 --quad_landmark_features --landmark_features data/misc/overhand_landmarks_training_70.h5 --rbf data/misc/overhand_actions_training_tfm_100.h5 data/misc/overhand_holdout_set.h5 data/weights/nearestneighbor_weights.h5 data/misc/fake_data_segment.h5
"""


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
    table_height = 0.
    rope_observe_cloud_upsample=0

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
    return [trajs, cost, xyz_src_warped, f]

def compute_trans_traj(sim_env, new_xyz, seg_info, ignore_infeasibility=True, animate=False, interactive=False):
    redprint("Generating end-effector trajectory")    
    
    
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    l1 = len(old_xyz)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    l2 = len(new_xyz)
    print l1, l2
    
            
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    if GlobalVars.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
    else:
        interest_pts = None
    lr2eetraj, _, old_xyz_warped, f = warp_hmats_tfm(old_xyz, new_xyz, hmat_list, interest_pts)


    handles = []
    if animate:
        handles.extend(plotting_openrave.draw_grid(sim_env.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))
        handles.append(sim_env.env.plot3(old_xyz,5, (1,0,0))) # red: demonstration point cloud
        handles.append(sim_env.env.plot3(new_xyz,5, (0,0,1))) # blue: rope nodes
        handles.append(sim_env.env.plot3(old_xyz_warped,5, (0,1,0))) # green: warped point cloud from demonstration
        
        mapped_pts = []
        for i in range(len(old_xyz)):
            mapped_pts.append(old_xyz[i])
            mapped_pts.append(old_xyz_warped[i])
        handles.append(sim_env.env.drawlinelist(np.array(mapped_pts), 1, [0.1,0.1,1]))
        
    for lr in 'lr':
        handles.append(sim_env.env.drawlinestrip(lr2eetraj[lr][:,:3,3], 2, (0,1,0,1)))
        
    for k, hmats in hmat_list:
        hmats_tfm = np.asarray([GlobalVars.init_tfm.dot(h) for h in hmats])
        handles.append(sim_env.env.drawlinestrip(hmats_tfm[:,:3,3], 2, (1,0,0,1)))
        

    miniseg_starts, miniseg_ends, lr_open = sim_util.split_trajectory_by_gripper(seg_info, thresh=GRIPPER_ANGLE_THRESHOLD)    
    success = True
    feasible = True
    misgrasp = False
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    miniseg_trajs = []
    prev_vals = {lr:None for lr in 'lr'}
    
    
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))


        ### adaptive resampling based on xyz in end_effector
        end_trans_trajs = np.zeros([i_end+1-i_start, 6])

        for lr in 'lr':
            for i in xrange(i_start,i_end+1):
                if lr == 'l':
                    end_trans_trajs[i-i_start, :3] = lr2eetraj[lr][i][:3,3]
                else:
                    end_trans_trajs[i-i_start, 3:] = lr2eetraj[lr][i][:3,3]

        if True:
            adaptive_times, end_trans_trajs = resampling.adaptive_resample2(end_trans_trajs, 0.005)
        else:
            adaptive_times = range(len(end_trans_trajs))
            

        miniseg_traj = {}
        for lr in 'lr':
            #ee_hmats = resampling.interp_hmats(np.arange(i_end+1-i_start), np.arange(i_end+1-i_start), lr2eetraj[lr][i_start:i_end+1])
            ee_hmats = resampling.interp_hmats(adaptive_times, np.arange(i_end+1-i_start), lr2eetraj[lr][i_start:i_end+1])
            
            # the interpolation above will then the velocity of the trajectory (since there are fewer waypoints). Resampling again to make sure 
            # the trajectory has the same number of waypoints as before.
            ee_hmats = resampling.interp_hmats(np.arange(i_end+1-i_start), adaptive_times, ee_hmats)
            
            # if arm_moved(ee_hmats, floating=True):
            if True:
                miniseg_traj[lr] = ee_hmats
                

        
        safe_drop = {'l': True, 'r': True}
        for lr in 'lr':
            next_gripper_open = lr_open[lr][i_miniseg+1] if i_miniseg < len(miniseg_starts) - 1 else False
            gripper_open = lr_open[lr][i_miniseg] 
            
            if next_gripper_open and not gripper_open:
                tfm = miniseg_traj[lr][-1]
                if tfm[2,3] > GlobalVars.table_height + 0.01:
                    safe_drop[lr] = False
                    
        #safe_drop = {'l': True, 'r': True}             
                
        if not (safe_drop['l'] and safe_drop['r']):
            for lr in 'lr':
                
                if not safe_drop[lr]:
                    tfm = miniseg_traj[lr][-1]
                    for i in range(1, 8):
                        safe_drop_tfm = tfm
                        safe_drop_tfm[2,3] = tfm[2,3] - i / 10. * (tfm[2,3] - GlobalVars.table_height - 0.01)
                        miniseg_traj[lr].append(safe_drop_tfm)
                else:
                    for i in range(1, 8):
                        miniseg_traj[lr].append(miniseg_traj[lr][-1])
                     
                
        miniseg_trajs.append(miniseg_traj)
        

        for lr in 'lr':
            hmats = np.asarray(miniseg_traj[lr])
            handles.append(sim_env.env.drawlinestrip(hmats[:,:3,3], 2, (0,0,1,1)))
            
        redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, miniseg_traj.keys()))
          
        
        for lr in 'lr':
            gripper_open = lr_open[lr][i_miniseg]
            prev_gripper_open = lr_open[lr][i_miniseg-1] if i_miniseg != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open, floating=True):
                redprint("Grab %s failed"%lr)
                success = False

        if not success: break
        
        if len(miniseg_traj) > 0:
            success &= sim_util.exec_traj_sim(sim_env, miniseg_traj, animate=animate)

        if not success: break

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, miniseg_trajs

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
        
    GlobalVars.rope_observe_cloud_upsample = 2

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
    
    parser_replay = subparsers.add_parser('playback')
    parser_replay.add_argument("--exec_rope_params", type=str, default='default')



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
        sim_util.reset_arms_to_side(sim_env, floating=True)
        redprint("Replace rope")
        rope_xyz = demo_id_rope_nodes["rope_nodes"][:]
        # Transform rope_nodes from the kinect's frame into the frame of the PR2
        if 'frame' not in demo_id_rope_nodes or demo_id_rope_nodes['frame'][()] != 'r':
            redprint("Transforming rope into frame of robot")
            rope_xyz = rope_xyz.dot(GlobalVars.init_tfm[:3,:3].T) + GlobalVars.init_tfm[:3,3][None,:]
        rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz)

        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env, rope_params=sim_util.get_rope_params("thick"), floating=True)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, args.exec_rope_params)

        for i_step in range(args.num_steps):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env, floating=True)

            redprint("Observe point cloud")
            new_xyz = sim_env.sim.observe_cloud(GlobalVars.rope_observe_cloud_upsample)
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
                    time_machine.restore_from_checkpoint(chkpt, sim_env, rope_params=sim_util.get_rope_params("thick"))
                    cur_xyz = sim_env.sim.observe_cloud(GlobalVars.rope_observe_cloud_upsample)
                    success, _, _, full_trajs = \
                        compute_trans_traj(sim_env, cur_xyz, GlobalVars.actions[a], animate=args.animation, interactive=False)
                    if args.animation:
                        sim_env.viewer.Step()
                    if is_knot(sim_env.sim.rope.GetControlPoints()):
                        best_root_action = r_a
                        break
                    result_cloud = sim_env.sim.observe_cloud(GlobalVars.rope_observe_cloud_upsample)
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
            
            time_machine.restore_from_checkpoint('depth_0_%i'%i_step, sim_env, rope_params=sim_util.get_rope_params("thick"))
            eval_stats = eval_util.EvalStats()
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = \
                compute_trans_traj(sim_env, new_xyz, GlobalVars.actions[best_root_action], animate=args.animation, interactive=args.interactive)
            
            print "BEST ACTION:", best_root_action
            eval_util.save_task_results_step(args.resultfile, sim_env, i_task, i_step, eval_stats, best_root_action, full_trajs, q_values_root, floating=True)
            
            if is_knot(sim_env.sim.rope.GetControlPoints()):
                redprint("KNOT TIED!")
                break;

        if is_knot(sim_env.sim.rope.GetControlPoints()):
            num_successes += 1
        num_total += 1

        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))
        
        
        
        
def eval_demo_playback(args, sim_env):
    actionfile = GlobalVars.actions
    num_successes = 0
    num_total = 0
    
    actions_names = []
    for action_name in actionfile:
        actions_names.append(action_name)
        
    i = 0
    i_task = 0
    i_step = 0
    while i < len(actions_names):
        (demo_name, seg_name) = actions_names[i].split('-')
        
        if seg_name == 'seg00':
            if i > 0:
                if is_knot(sim_env.sim.rope.GetControlPoints()):
                    redprint("KNOT TIED!")
                    num_successes += 1
                num_total += 1
            
            i_task += 1
            i_step = 0
            print "task %s" % demo_name
            sim_util.reset_arms_to_side(sim_env, floating=True)
            redprint("Replace rope")
            rope_xyz = np.asarray(actionfile[actions_names[i]]['cloud_xyz'])
            rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz)
            
            # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
            time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env, rope_params=sim_util.get_rope_params("thick"), floating=True)
        
            if args.animation:
                sim_env.viewer.Step()
                    
        print "task %s step %i" % (i_task, i_step)
        sim_util.reset_arms_to_side(sim_env, floating=True)
            
        redprint("Observe point cloud")
        #new_xyz = sim_env.sim.observe_cloud(GlobalVars.rope_observe_cloud_upsample)
        new_xyz = sim_env.sim.observe_cloud2(0.01, GlobalVars.rope_observe_cloud_upsample, 1, 4)
        state = ("eval_%i"%get_unique_id(), new_xyz) 
        
        action = actions_names[i]
        
        eval_stats = eval_util.EvalStats()
        eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = \
        compute_trans_traj(sim_env, new_xyz, GlobalVars.actions[action], animate=args.animation, interactive=args.interactive)
                        
        i += 1
        
    num_total += 1

    redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))        

# make args more module (i.e. remove irrelevant args for replay mode)
# TODO: Make this work with floating grippers
def replay_on_holdout(args, sim_env):
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    tasks = eval_util.get_specified_tasks(args.tasks, args.taskfile, args.i_start, args.i_end)
    loadresultfile = h5py.File(args.loadresultfile, 'r')
    loadresult_items = eval_util.get_holdout_items(loadresultfile, tasks)

    num_successes = 0
    num_total = 0
    
    for i_task, demo_id_rope_nodes in loadresult_items:
        print "task %s" % i_task
        sim_util.reset_arms_to_side(sim_env, floating=True)
        redprint("Replace rope")
        rope_nodes, rope_params, _, _ = eval_util.load_task_results_init(args.loadresultfile, i_task)
        # uncomment if the results file don't have the right rope nodes
        #rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        if args.replay_rope_params:
            rope_params = args.replay_rope_params
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env, floating=True)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, rope_params)

        for i_step in range(len(loadresultfile[i_task]) - (1 if 'init' in loadresultfile[i_task] else 0)):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env, floating=True)

            redprint("Observe point cloud")
            new_xyz = sim_env.sim.observe_cloud(GlobalVars.rope_observe_cloud_upsample)
    
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

    actions = h5py.File(args.actionfile, 'r')
    fakedatafile = h5py.File(args.fakedatafile, 'r')
    GlobalVars.init_tfm = fakedatafile['init_tfm'][()]
    
    init_rope_xyz, _ = sim_util.load_fake_data_segment(sim_env, fakedatafile, args.fake_data_segment, args.fake_data_transform, floating=True) # this also sets the torso (torso_lift_joint) to the height in the data
    GlobalVars.init_tfm = GlobalVars.init_tfm
    

    
    # Set table height to correct height of first rope in holdout set
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    first_holdout = holdoutfile[holdoutfile.keys()[0]]
    init_rope_xyz = first_holdout['rope_nodes'][:]
    if 'frame' not in first_holdout or first_holdout['frame'][()] != 'r':
        init_rope_xyz = init_rope_xyz.dot(GlobalVars.init_tfm[:3,:3].T) + GlobalVars.init_tfm[:3,3][None,:]


    table_height = init_rope_xyz[:,2].mean() - .02 # Before: .02
    GlobalVars.table_height = table_height
    table_xml = sim_util.make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    sim_env.env.LoadData(table_xml)

    sim_util.reset_arms_to_side(sim_env, floating=True)
    
    if args.animation:
        sim_env.viewer = trajoptpy.GetViewer(sim_env.env)
        table = sim_env.env.GetKinBody('table')
        sim_env.viewer.SetTransparency(table, 0.4)
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
    elif args.subparser_name == "playback":
        eval_demo_playback(args, sim_env)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
