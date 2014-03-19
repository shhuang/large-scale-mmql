#!/usr/bin/env python

import pprint
import argparse

from rapprentice import registration, colorize, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     planning, tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time
import ropesim_floating

try:
    from rapprentice import pr2_trajectories, PR2
    import rospy
except ImportError:
    print "Couldn't import ros stuff"

import cloudprocpy, trajoptpy, openravepy
import util
from rope_qlearn import *
from knot_classifier import isKnot as is_knot
import os, numpy as np, h5py
from numpy import asarray
from numpy.linalg import norm
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random

"""
Transform from lr_gripper_tool_frame to end_effector_transform.
This is so that you can give openrave the data in the frame it is expecting.
Openrave does IK in end_effector_frame which is different from gripper_tool_frame.
"""
TFM_GTF_EE = np.array([[ 0.,  0.,  1.,  0.],
                       [ 0.,  1.,  0.,  0.],
                       [-1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.]])

L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)

GRIPPER_ANGLE_THRESHOLD = 10.0 # Open/close threshold, adjusted for human demonstrations

def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)

def blueprint(msg):
    print colorize.colorize(msg, "blue", bold=True)

def yellowprint(msg):
    print colorize.colorize(msg, "yellow", bold=True)

def split_trajectory_by_gripper(seg_info, ms_thresh=2):
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

    #thresh = .04 # open/close threshold
    thresh = GRIPPER_ANGLE_THRESHOLD # open/close threshold

    n_steps = len(lgrip)


    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= thresh) & (lgrip[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip[1:] >= thresh) & (rgrip[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip[1:] < thresh) & (lgrip[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip[1:] < thresh) & (rgrip[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])
    
    lr_open = {lr:[] for lr in 'lr'}
    new_seg_starts = []
    new_seg_ends = []
    for i in range(len(seg_starts)):
        if seg_ends[i]- seg_starts[i] >= ms_thresh:
            new_seg_starts.append(seg_starts[i])
            new_seg_ends.append(seg_ends[i])
            lval = True if lgrip[seg_starts[i]] >= thresh else False
            lr_open['l'].append(lval)
            rval = True if rgrip[seg_starts[i]] >= thresh else False
            lr_open['r'].append(rval)

    return new_seg_starts, new_seg_ends, lr_open

def binarize_gripper(angle):
    #thresh = .04
    thresh = GRIPPER_ANGLE_THRESHOLD
    return angle > thresh



def set_gripper_sim(lr, is_open, prev_is_open):
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

    # release constraints if necessary
    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)
        print "DONE RELEASING"

    # execute gripper open/close trajectory
    start_val = Globals.sim.grippers[lr].get_gripper_joint_value()
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        Globals.sim.grippers[lr].set_gripper_joint_value(val)
        Globals.sim.step()
        #if Globals.viewer:
        #    Globals.viewer.Step()
        #    if args.interactive: Globals.viewer.Idle()
    # add constraints if necessary
    if not is_open and prev_is_open:
        if not Globals.sim.grab_rope(lr):
            return False
    return True

def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2,4,6]:
        traj[:,i] = np.unwrap(traj[:,i])
    return traj

def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:,:7])
        unwrap_arm_traj_in_place(t[:,7:])
    else:
        raise NotImplementedError
    
    
def animate_floating_traj(lhmats, rhmats, sim, pause=True, step_viewer=True, callback=None,step=5):
    assert len(lhmats)==len(rhmats), "I don't know how to animate trajectory with different lengths"
    if step_viewer or pause: viewer = trajoptpy.GetViewer(sim.env)
    for i in xrange(len(lhmats)):
        if callback is not None: callback(i)
        sim.grippers['r'].set_toolframe_transform(rhmats[i])
        sim.grippers['l'].set_toolframe_transform(lhmats[i])
        if pause: viewer.Idle()
        elif step_viewer and not i%step: viewer.Step()


def exec_traj_sim(lr_traj, animate=False, interactive=False, step=1):
    def sim_callback(i):
        Globals.sim.step()

    lhmats_up, rhmats_up = ropesim_floating.retime_hmats(lr_traj['l'], lr_traj['r'])

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_rtf  = Globals.sim.grippers['r'].get_toolframe_transform()
    curr_ltf  = Globals.sim.grippers['l'].get_toolframe_transform()

    l_transition_hmats, r_transition_hmats = ropesim_floating.retime_hmats([curr_ltf, lhmats_up[0]], [curr_rtf, rhmats_up[0]])

    animate_floating_traj(l_transition_hmats, r_transition_hmats,
                          Globals.sim, pause=False,
                          callback=sim_callback, step_viewer=animate, step=step)
    animate_floating_traj(lhmats_up, rhmats_up, Globals.sim, pause=False,
                          callback=sim_callback, step_viewer=animate, step=step)
    return True

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def sample_rope_state(demofile, human_check=True, perturb_points=5, min_rad=0, max_rad=.15):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        new_xyz= load_random_start_segment(demofile)
        perturb_radius = random.uniform(min_rad, max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                        perturb_peak_dist=perturb_radius,
                                                                        num_perturb_points=perturb_points)
        replace_rope(rope_nodes)
        Globals.sim.settle()
        Globals.viewer.Step()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True

DS_SIZE = .025

def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def smaller_ang(x):
    return (x + np.pi)%(2*np.pi) - np.pi

def closer_ang(x,a,dr=0):
    """                                                
    find angle y (==x mod 2*pi) that is close to a
    dir == 0: minimize absolute value of difference
    dir == 1: y > x
    dir == 2: y < x
    """
    if dr == 0:
        return a + smaller_ang(x-a)
    elif dr == 1:
        return a + (x-a)%(2*np.pi)
    elif dr == -1:
        return a + (x-a)%(2*np.pi) - 2*np.pi

def closer_angs(x_array,a_array,dr=0):
    return [closer_ang(x, a, dr) for (x, a) in zip(x_array, a_array)]

def lerp (x, xp, fp, first=None):
    """
    Returns linearly interpolated n-d vector at specified times.
    """

    fp = np.asarray(fp)
    fp_interp = np.empty((len(x),0))
    for idx in range(fp.shape[1]):
        if first is None:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx])).T
        else:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx],left=first[idx])).T
        fp_interp = np.c_[fp_interp, interp_vals]

    return fp_interp

def close_traj(traj):
    assert len(traj) > 0

    curr_angs = traj[0]
    new_traj = []
    for i in xrange(len(traj)):
        new_angs = traj[i]
        for j in range(len(new_angs)):
            new_angs[j] = closer_ang(new_angs[j], curr_angs[j])
        new_traj.append(new_angs)
        curr_angs = new_angs

    return new_traj

def warp_hmats_tfm(xyz_src, xyz_targ, hmat_list, src_interest_pts = None):
    f, src_params, g, targ_params, cost = registration_cost(xyz_src, xyz_targ, src_interest_pts)
    f = registration.unscale_tps(f, src_params, targ_params)
    trajs = {}
    xyz_src_warped = np.zeros(xyz_src.shape)
    for k, hmats in hmat_list:
        # First transform hmats from the camera frame into the frame of the robot
        hmats_tfm = np.asarray([Globals.init_tfm.dot(h) for h in hmats])
        trajs[k] = f.transform_hmats(hmats_tfm)
    xyz_src_warped = f.transform_points(xyz_src)
    return [trajs, cost, xyz_src_warped]

def simulate_demo(new_xyz, seg_info, animate=False):    
    redprint("Generating end-effector trajectory")    
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
    handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    if args.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
    else:
        interest_pts = None
    lr2eetraj = warp_hmats_tfm(old_xyz, new_xyz, hmat_list, interest_pts)[0]

    miniseg_starts, miniseg_ends, lr_open = split_trajectory_by_gripper(seg_info)    
    success = True
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    miniseg_trajs = []
    prev_vals = {lr:None for lr in 'lr'}
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))
        
        
        miniseg_traj = {}
        for lr in 'lr':
            ee_hmats = resampling.interp_hmats(np.arange(i_end+1-i_start), np.arange(i_end+1-i_start), lr2eetraj[lr][i_start:i_end+1])
            # if arm_moved(ee_hmats):
            if True:
                miniseg_traj[lr] = ee_hmats;
                
        miniseg_trajs.append(miniseg_traj);

        redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, miniseg_traj.keys()))
        
        for lr in 'lr':
            gripper_open = lr_open[lr][i_miniseg]
            prev_gripper_open = lr_open[lr][i_miniseg-1] if i_miniseg != 0 else False
            if not set_gripper_sim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed"%lr)
                success = False

        if not success: break
        
        if len(miniseg_traj) > 0:
            success &= exec_traj_sim(miniseg_traj, animate=animate)

        if not success: break

    Globals.sim.settle(animate=animate)
    if animate:
        Globals.viewer.Step()
    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    
    return success, miniseg_trajs


def replace_rope(new_rope):
    import bulletsimpy
    old_rope_nodes = Globals.sim.rope.GetControlPoints()
    if Globals.viewer:
        Globals.viewer.RemoveKinBody(Globals.env.GetKinBody('rope'))
    Globals.env.Remove(Globals.env.GetKinBody('rope'))
    Globals.sim.bt_env.Remove(Globals.sim.bt_env.GetObjectByName('rope'))
    Globals.sim.rope = bulletsimpy.CapsuleRope(Globals.sim.bt_env, 'rope', new_rope,
                                               Globals.sim.rope_params)
    return old_rope_nodes

def get_rope_transforms():
    return (Globals.sim.rope.GetTranslations(), Globals.sim.rope.GetRotations())    

def set_rope_transforms(tfs):
    Globals.sim.rope.SetTranslations(tfs[0])
    Globals.sim.rope.SetRotations(tfs[1])

def arm_moved(hmat_traj):
    if len(hmat_traj) < 2:
        return False
    tts = hmat_traj[:,:3,3]
    return ((tts[1:] - tts[:-1]).ptp(axis=0) > .01).any()

        
def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0,1)))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if Globals.viewer:
        Globals.viewer.Step()

def load_fake_data_segment(demofile, fake_data_segment, fake_data_transform, set_robot_state=True):
    fake_seg = demofile[fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])
    hmat[:3,3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
    return new_xyz

def unif_resample(traj, max_diff, wt = None):        
    """
    Resample a trajectory so steps have same length in joint space    
    """
    import scipy.interpolate as si
    tol = .005
    if wt is not None: 
        wt = np.atleast_2d(wt)
        traj = traj*wt
        
        
    dl = mu.norms(traj[1:] - traj[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1: return traj, np.arange(len(traj))
    
    nsteps = max(int(np.ceil(float(l[-1])/max_diff)), 2)
    newl = np.linspace(0,l[-1],nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps,ncols)) 
    for istart in xrange(0, traj.shape[1], colstep):
        (tck,_) = si.splprep(traj[goodinds, istart:istart+colstep].T,k=deg,s = tol**2*len(traj),u=l[goodinds])
        traj_rs[:,istart:istart+colstep] = np.array(si.splev(newl,tck)).T
    if wt is not None: traj_rs = traj_rs/wt

    newt = np.interp(newl, l, np.arange(len(traj)))

    return traj_rs, newt

def make_table_xml(translation, extents):
    xml = """
<Environment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

L_POSTURES = {'side': np.array([[-0.98108876, -0.1846131 ,  0.0581623 ,  0.10118172],
                                [-0.19076337,  0.97311662, -0.12904799,  0.68224057],
                                [-0.03277475, -0.13770277, -0.98993119,  0.91652485],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

R_POSTURES = {'side' : np.array([[-0.98108876,  0.1846131 ,  0.0581623 ,  0.10118172],
                                 [ 0.19076337,  0.97311662,  0.12904799, -0.68224057],
                                 [-0.03277475,  0.13770277, -0.98993119,  0.91652485],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]) }



def reset_arms_to_side():
    Globals.sim.grippers['r'].set_toolframe_transform(R_POSTURES['side'])
    Globals.sim.grippers['l'].set_toolframe_transform(L_POSTURES['side'])

###################

class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    log = None
    viewer = None
    resample_rope = None
    init_tfm = None

if __name__ == "__main__":
    """
    example command:
    ./do_task_eval.py data/weights/multi_quad_weights_10000.h5 --quad_features --animation=1
    
    ./do_task_eval.py --resultfile test_results.h5 --animation 1 --quad_landmark_features --landmark_features data/misc/overhand_landmarks_training_70.h5 --rbf data/misc/overhand_actions_training_tfm_100.h5 data/misc/overhand_holdout_set.h5 data/weights/nearestneighbor_weights.h5 data/misc/fake_data_segment.h5

    """
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser.add_argument('holdoutfile', nargs='?', default='data/misc/holdout_set.h5')
    parser.add_argument("weightfile", type=str)
    parser.add_argument("fakedatafile", type=str) # required since large-scale actions file does not contain robot positions
    parser.add_argument("--resultfile", type=str) # don't save results if this is not specified
    parser.add_argument("--lookahead_width", type=int, default=1)
    parser.add_argument('--lookahead_depth', type=int, default=0)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--rbf', action='store_true')
    parser.add_argument('--landmark_features')
    parser.add_argument('--quad_landmark_features', action='store_true')
    parser.add_argument('--only_landmark', action="store_true")
    parser.add_argument("--quad_features", action="store_true")
    parser.add_argument("--sc_features", action="store_true")
    parser.add_argument("--rope_dist_features", action="store_true")
    parser.add_argument("--traj_features", action="store_true")
    parser.add_argument("--gripper_weighting", action="store_true")
    parser.add_argument("--animation", type=int, default=0)
    parser.add_argument("--i_start", type=int, default=-1)
    parser.add_argument("--i_end", type=int, default=-1)
    
    parser.add_argument("--tasks", nargs='+', type=int)
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--num_steps", type=int, default=5)
    
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--log", type=str, default="", help="")
    
    args = parser.parse_args()

    if args.random_seed is not None: np.random.seed(args.random_seed)

    trajoptpy.SetInteractive(args.interactive)

    if args.log:
        redprint("Writing log to file %s" % args.log)
        Globals.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(Globals.exec_log.close)
        Globals.exec_log(0, "main.args", args)

    Globals.env = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.sim = ropesim_floating.FloatingGripperSimulation(Globals.env)

    actionfile = h5py.File(args.actionfile, 'r')
    fakedatafile = h5py.File(args.fakedatafile, 'r')
    Globals.init_tfm = fakedatafile['init_tfm'][()]
    
    init_rope_xyz = load_fake_data_segment(fakedatafile, args.fake_data_segment, args.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data

    # Set table height to correct height of first rope in holdout set
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    first_holdout = holdoutfile[holdoutfile.keys()[0]]
    init_rope_xyz = first_holdout['rope_nodes'][:]
    if 'frame' not in first_holdout or first_holdout['frame'][()] != 'r':
        init_rope_xyz = init_rope_xyz.dot(Globals.init_tfm[:3,:3].T) + Globals.init_tfm[:3,3][None,:]

    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    Globals.env.LoadData(table_xml)
    # create rope from rope in data
    rope_nodes = rope_initialization.find_path_through_point_cloud(init_rope_xyz)
    Globals.sim.create(rope_nodes)
    # move arms to the side
    reset_arms_to_side()

    if args.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
        print "move viewer to viewpoint that isn't stupid"
        print "then hit 'p' to continue"
        Globals.viewer.Idle()

    #####################
    feature_fn, _, num_features, actions = select_feature_fn(args)

    weightfile = h5py.File(args.weightfile, 'r')
    weights = weightfile['weights'][:]
    w0 = weightfile['w0'][()] if 'w0' in weightfile else 0
    weightfile.close()
    assert weights.shape[0] == num_features, "Dimensions of weights and features don't match. Make sure the right feature is being used"
    
    save_results = args.resultfile is not None
    
    unique_id = 0
    def get_unique_id():
        global unique_id
        unique_id += 1
        return unique_id-1

    tasks = [] if args.tasks is None else args.tasks
    if args.taskfile is not None:
        file = open(args.taskfile, 'r')
        for line in file.xreadlines():
            tasks.append(int(line[5:-1]))
    if args.i_start != -1 and args.i_end != -1:
        tasks = range(args.i_start, args.i_end)

    def q_value_fn(state, action):
        return np.dot(weights, feature_fn(state, action)) + w0
    def value_fn(state):
        state = state[:]
        return max(q_value_fn(state, action) for action in actions)

    for i_task, demo_id_rope_nodes in (holdoutfile.iteritems() if not tasks else [(unicode(t),holdoutfile[unicode(t)]) for t in tasks]):
        reset_arms_to_side()

        redprint("Replace rope")
        rope_xyz = demo_id_rope_nodes["rope_nodes"][:]
        # Transform rope_nodes from the kinect's frame into the frame of the PR2
        if 'frame' not in demo_id_rope_nodes or demo_id_rope_nodes['frame'][()] != 'r':
            redprint("Transforming rope into frame of robot")
            rope_xyz = rope_xyz.dot(Globals.init_tfm[:3,:3].T) + Globals.init_tfm[:3,3][None,:]
        rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz)

        # TODO: Remove after debugging
        #if sum(rope_nodes[:,2] < table_height) > 0:
        #    print sum(rope_nodes[:,2] < table_height)
        #    raw_input('Press <ENTER>')

        replace_rope(rope_nodes)
        Globals.sim.settle()
        if args.animation:
            Globals.viewer.Step()

        if save_results:
            result_file = h5py.File(args.resultfile, 'a')
            if i_task in result_file:
                del result_file[i_task]
            result_file.create_group(i_task)
        
        for i_step in range(args.num_steps):
            print "task %s step %i" % (i_task, i_step)

            reset_arms_to_side()

            redprint("Observe point cloud")
            new_xyz = Globals.sim.observe_cloud()
            state = ("eval_%i"%get_unique_id(), new_xyz)
    

            Globals.sim.observe_cloud()
            if is_knot(Globals.sim.observe_cloud()):
                redprint("KNOT TIED!")
                break;

            redprint("Choosing an action")
            q_values = [q_value_fn(state, action) for action in actions]
            q_values_root = q_values
            rope_tf = get_rope_transforms()

            assert args.lookahead_width>= 1, 'Lookahead branches set to zero will fail to select any action'
            agenda = sorted(zip(q_values, actions), key = lambda v: -v[0])[:args.lookahead_width]
            agenda = [(v, a, rope_tf, a) for (v, a) in agenda] # state is (value, most recent action, rope_transforms, root action)
            best_root_action = None
            for _ in range(args.lookahead_depth):
                expansion_results = []
                for (q, a, tf, r_a) in agenda:
                    set_rope_transforms(tf)                 
                    cur_xyz = Globals.sim.observe_cloud()
                    success, ee_trajs = simulate_demo(cur_xyz, actionfile[a], animate=False)
                    if args.animation:
                        Globals.viewer.Step()
                    result_cloud = Globals.sim.observe_cloud()
                    if is_knot(result_cloud):
                        best_root_action = r_a
                        break
                    expansion_results.append((result_cloud, a, success, get_rope_transforms(), r_a))
                if best_root_action is not None:
                    redprint('Knot Found, stopping search early')
                    break
                agenda = []
                for (cld, incoming_a, success, tf, r_a) in expansion_results:
                    if not success:
                        agenda.append((-np.inf, actions[0], tf, r_a))
                        continue
                    next_state = ("eval_%i"%get_unique_id(), cld)
                    q_values = [(q_value_fn(next_state, action), action, tf, r_a) for action in actions]
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
            set_rope_transforms(rope_tf) # reset rope to initial state
            print "BEST ROOT ACTION:", best_root_action

            #continue # TODO: Remove after debugging

            success, trajs = simulate_demo(new_xyz, actionfile[best_root_action], animate=args.animation)
            set_rope_transforms(get_rope_transforms())
            
            if save_results:
                result_file[i_task].create_group(str(i_step))
                result_file[i_task][str(i_step)]['rope_nodes'] = Globals.sim.rope.GetControlPoints()
                result_file[i_task][str(i_step)]['best_action'] = str(best_root_action)
                trajs_g = result_file[i_task][str(i_step)].create_group('trajs')
                for (i_traj,traj) in enumerate(trajs):
                    traj_g = trajs_g.create_group(str(i_traj))
                    for (ee_name, ee_traj) in traj.iteritems():
                        traj_g[str(ee_name)] = ee_traj
                result_file[i_task][str(i_step)]['values'] = q_values_root
        if save_results:
            result_file.close()
