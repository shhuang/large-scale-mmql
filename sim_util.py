# Contains useful functions for PR2 rope tying simulation
# The purpose of this class is to eventually consolidate
# the various instantiations of do_task_eval.py

import h5py
import bulletsimpy
import openravepy, trajoptpy
import numpy as np
from numpy import asarray
import re

from rapprentice import animate_traj, ropesim, ros2rave, math_utils as mu

PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)

class SimulationEnv:
    def __init__(self):
        self.robot = None
        self.env = None
        self.pr2 = None
        self.sim = None
        self.log = None
        self.viewer = None

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

def make_box_xml(name, translation, extents):
    xml = """
<Environment>
  <KinBody name="%s">
    <Body type="dynamic" name="%s_link">
      <Translation>%f %f %f</Translation>
      <Geom type="box">
        <extents>%f %f %f</extents>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (name, name, translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

def make_cylinder_xml(name, translation, radius, height):
    xml = """
<Environment>
  <KinBody name="%s">
    <Body type="dynamic" name="%s_link">
      <Translation>%f %f %f</Translation>
      <Geom type="cylinder">
        <rotationaxis>1 0 0 90</rotationaxis>
        <radius>%f</radius>
        <height>%f</height>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (name, name, translation[0], translation[1], translation[2], radius, height)
    return xml

def reset_arms_to_side(sim_env):
    sim_env.robot.SetDOFValues(PR2_L_POSTURES["side"],
                               sim_env.robot.GetManipulator("leftarm").GetArmIndices())
    #actionfile = None
    sim_env.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               sim_env.robot.GetManipulator("rightarm").GetArmIndices())
    mult = 5
    open_angle = .08 * mult
    for lr in 'lr':
        joint_ind = sim_env.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = sim_env.robot.GetDOFValues([joint_ind])[0]
        sim_env.robot.SetDOFValues([open_angle], [joint_ind])

def arm_moved(joint_traj):    
    if len(joint_traj) < 2: return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()

def split_trajectory_by_gripper(seg_info, thresh = .04):
    # thresh: open/close threshold
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

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

    return seg_starts, seg_ends

def binarize_gripper(angle, thresh = .04):
    return angle > thresh
    
def set_gripper_maybesim(sim_env, lr, is_open, prev_is_open):
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle
    
    # release constraints if necessary
    if is_open and not prev_is_open:
        sim_env.sim.release_rope(lr)
        print "DONE RELEASING"

    # execute gripper open/close trajectory
    joint_ind = sim_env.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
    start_val = sim_env.robot.GetDOFValues([joint_ind])[0]
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        sim_env.robot.SetDOFValues([val], [joint_ind])
        sim_env.sim.step()
#         if args.animation:
#                sim_env.viewer.Step()
#             if args.interactive: sim_env.viewer.Idle()
    # add constraints if necessary
    if sim_env.viewer:
        sim_env.viewer.Step()
    if not is_open and prev_is_open:
        if not sim_env.sim.grab_rope(lr):
            return False

    return True

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

def sim_traj_maybesim(sim_env, bodypart2traj, animate=False, interactive=False):
    full_traj = getFullTraj(sim_env, bodypart2traj)
    return sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive)

def sim_full_traj_maybesim(sim_env, full_traj, animate=False, interactive=False):
    def sim_callback(i):
        sim_env.sim.step()

    animate_speed = 10 if animate else 0

    traj, dof_inds = full_traj

    # make the trajectory slow enough for the simulation
    traj = ropesim.retime_traj(sim_env.robot, dof_inds, traj)

    # in simulation mode, we must make sure to gradually move to the new starting position
    sim_env.robot.SetActiveDOFs(dof_inds)
    curr_vals = sim_env.robot.GetActiveDOFValues()
    transition_traj = np.r_[[curr_vals], [traj[0]]]
    unwrap_in_place(transition_traj)
    transition_traj = ropesim.retime_traj(sim_env.robot, dof_inds, transition_traj, max_cart_vel=.05)
    animate_traj.animate_traj(transition_traj, sim_env.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    traj[0] = transition_traj[-1]
    unwrap_in_place(traj)

    animate_traj.animate_traj(traj, sim_env.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    if sim_env.viewer:
        sim_env.viewer.Step()
    return True

def getFullTraj(sim_env, bodypart2traj):
    """
    A full trajectory is a tuple of a trajectory (np matrix) and dof indices (list)
    """
    if len(bodypart2traj) > 0:
        trajs = []
        dof_inds = []
        for (part_name, traj) in bodypart2traj.items():
            manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
            trajs.append(traj)
            dof_inds.extend(sim_env.robot.GetManipulator(manip_name).GetArmIndices())            
        full_traj = (np.concatenate(trajs, axis=1), dof_inds)
    else:
        full_traj = (np.zeros((0,0)), [])
    return full_traj

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def load_fake_data_segment(sim_env, demofile, fake_data_segment, fake_data_transform, set_robot_state=True):
    fake_seg = demofile[fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])
    hmat[:3,3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
    r2r = ros2rave.RosToRave(sim_env.robot, asarray(fake_seg["joint_states"]["name"]))
    if set_robot_state:
        r2r.set_values(sim_env.robot, asarray(fake_seg["joint_states"]["position"][0]))
    return new_xyz, r2r

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

def get_rope_transforms(sim_env):
    return (sim_env.sim.rope.GetTranslations(), sim_env.sim.rope.GetRotations())    

def replace_rope(new_rope, sim_env, rope_params=None):
    if sim_env.sim:
        for lr in 'lr':
            sim_env.sim.release_rope(lr)
    rope_kin_body = sim_env.env.GetKinBody('rope')
    if rope_kin_body:
        if sim_env.viewer:
            sim_env.viewer.RemoveKinBody(rope_kin_body)
    if sim_env.sim:
        del sim_env.sim
    sim_env.sim = ropesim.Simulation(sim_env.env, sim_env.robot, rope_params)
    sim_env.sim.create(new_rope)

def set_rope_transforms(tfs, sim_env):
    sim_env.sim.rope.SetTranslations(tfs[0])
    sim_env.sim.rope.SetRotations(tfs[1])

def get_rope_params(params_id):
    rope_params = bulletsimpy.CapsuleRopeParams()
    if params_id == 'default':
        rope_params.radius = 0.005
        rope_params.angStiffness = .1
        rope_params.angDamping = 1
        rope_params.linDamping = .75
        rope_params.angLimit = .4
        rope_params.linStopErp = .2
    elif params_id == 'thick':
        rope_params.radius = 0.008
        rope_params.angStiffness = .1
        rope_params.angDamping = 1
        rope_params.linDamping = .75
        rope_params.angLimit = .4
        rope_params.linStopErp = .2
    elif params_id.startswith('stiffness'):
        try:
            stiffness = float(re.search(r'stiffness(.*)', params_id).group(1))
        except:
            raise RuntimeError("Invalid rope parameter id")
        rope_params.radius = 0.005
        rope_params.angStiffness = stiffness
        rope_params.angDamping = 1
        rope_params.linDamping = .75
        rope_params.angLimit = .4
        rope_params.linStopErp = .2
    else:
        raise RuntimeError("Invalid rope parameter id")
    return rope_params

class RopeSimTimeMachine(object):
    """
    Sets and tracks the state of the rope in a consistent manner.
    Keeps track of the state of the rope at user-defined checkpoints and allows 
    for restoring from that checkpoint in a deterministic manner (i.e. calling
    time_machine.restore_from_checkpoint(id) should restore the same simulation
    state everytime it is called)
    """
    def __init__(self, new_rope, sim_env, rope_params=None):
        """
        new_rope is the initial rope_nodes of the machine for a particular task
        """
        self.rope_nodes = new_rope
        self.checkpoints = {}
        replace_rope(self.rope_nodes, sim_env, rope_params)
        sim_env.sim.settle()
        
    def set_checkpoint(self, id, sim_env, tfs=None):
        if id in self.checkpoints:
            raise RuntimeError("Can not set checkpoint with id %s since it has already been set"%id)
        if tfs:
            self.checkpoints[id] = tfs
        else:
            self.checkpoints[id] = get_rope_transforms(sim_env)

    def restore_from_checkpoint(self, id, sim_env, rope_params=None):
        if id not in self.checkpoints:
            raise RuntimeError("Can not restore checkpoint with id %s since it has not been set"%id)
        replace_rope(self.rope_nodes, sim_env, rope_params)
        set_rope_transforms(self.checkpoints[id], sim_env)
        sim_env.sim.settle()

def tpsrpm_plot_cb(sim_env, x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(sim_env.env.plot3(ypred_nd, 3, (0,1,0,1)))
    handles.extend(plotting_openrave.draw_grid(sim_env.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if sim_env.viewer:
        sim_env.viewer.Step()

