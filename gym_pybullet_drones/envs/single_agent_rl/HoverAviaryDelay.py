import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gymnasium import spaces
from gym_pybullet_drones.envs.single_agent_rl import HoverAviary
from collections import deque
import tensorboard

MAX_LIN_VEL_XY = 3
MAX_LIN_VEL_Z = 1

MAX_XY = 1.
MAX_Z = 1.
RPM_FACTOR = 0.2

class HoverAviaryDelay(HoverAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.TRPY,
                 add_action_obs = True,
                 add_pd = True
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        np.random.seed(0)
        self.goal = np.array([0., 0., 1.])
        if initial_xyzs is None:
            initial_xyzs = np.random.normal(0., 0.02, [3,])
            initial_xyzs = self.goal + initial_xyzs
            initial_xyzs = initial_xyzs[np.newaxis, :]
        if initial_rpys is None:
            initial_rp = np.random.normal(0., 5./180.*np.pi, [2,])
            initial_y = np.random.normal(0., 60./180. *np.pi,)
            initial_rpys = np.hstack([initial_rp, initial_y])
            initial_rpys = initial_rpys[np.newaxis, :]
        self.add_action_obs = add_action_obs
        self.add_pd = add_pd
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        # self.curriculum_stage = curriculum_stage
        self.EPISODE_LEN_SEC = 2
        self.time_constant = 0.05 # RPM time const
        self._reset_rpm()
        # self.control_history_steps_in_obs = 10

        self._reset_hist()
        self.rew_info = {}

    def _reset_rpm(self, seed=None):
        if seed:
            np.random.seed(seed)
        init_rpm_normalized = np.clip(np.random.normal(0., 0.3, size=[1, 4]), -1, 1)
        self.rpm = self.HOVER_RPM * (1 + 0.05 * init_rpm_normalized)

    def _reset_hist(self, hist_horizon = 36):
        self.pos_error = deque(maxlen=hist_horizon)
        for _ in range(hist_horizon):
            self.pos_error.append(np.zeros([3]))
        self.ang_error = deque(maxlen=hist_horizon)
        for _ in range(hist_horizon):
            self.ang_error.append(np.zeros([3]))
    ################################################################################

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        obs = self._clipAndNormalizeState(state)
        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # return obs
        ############################################################
        #### OBS SPACE OF SIZE 16 xyz 3, quat 4, rpy, vel_xyz, angle_vel_xyz 3 each
        # ret = np.hstack([obs[0:10], obs[10:13], obs[13:16]]).reshape(12, )
        ret = obs
        if self.add_pd:
            self.pos_error.append(state[0:3] - self.goal)
            self.ang_error.append(state[7:10])
            pos_error_i = np.array(self.pos_error).sum(axis=0) / (MAX_XY * len(self.pos_error))
            pos_error_d = self.pos_error[-1] - self.pos_error[-2] / (MAX_XY * 2)
            ang_error_i = np.array(self.ang_error).sum(axis=0) / (np.pi * len(self.ang_error))
            ang_error_d = self.ang_error[-1] - self.ang_error[-2] / (np.pi * 2)
            return np.hstack([ret, pos_error_i, pos_error_d, ang_error_i, ang_error_d]).astype('float32')
        else:
            return ret.astype('float32')

    def _observationSpace(self):
        return spaces.Box(low=-1 * np.ones([16 + 4 + 12 if self.add_pd else 16 + 4]),
                          high=np.ones([16 + 4 + 12 if self.add_pd else 16 + 4]),
                          dtype=np.float32
                          )

    
    def _computeReward(self, verbose=False):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # if self._computeTerminated():
        #     return -100.
        # if self.curriculum_stage == 1:
        #     return np.exp(-1. * np.linalg.norm(self.goal-state[0:3]) - 0.01 * state[9] ** 2 )   # - 1 * np.linalg.norm(state[13:16])
        # elif self.curriculum_stage == 2:
        rew_pos = - 2.5 * np.linalg.norm(self.goal-state[0:3])
        rew_rpy = - 1.5 * np.linalg.norm(state[7:9])
        rew_lin_vel = - 0.05 * np.linalg.norm(state[10:13])
        rew_ang_vel = - 0.05 * np.linalg.norm(state[13:16])
        rew_action = - 0.1 * np.linalg.norm(self.last_clipped_action[0] / self.MAX_RPM)
        rew_action_diff = -1. * np.linalg.norm((self.last_clipped_action[0] - self.last_step_action) / (2 * RPM_FACTOR * self.HOVER_RPM))
        self.rew_info = {'rew_pos': rew_pos,
                         'rew_rpy': rew_rpy,
                         'rew_lin_vel': rew_lin_vel,
                         'rew_ang_vel': rew_ang_vel,
                         'rew_action': rew_action,
                         'rew_action_diff': rew_action_diff}
        return 2 + (rew_pos +
                    rew_rpy +
                    rew_lin_vel +
                    rew_ang_vel +
                    rew_action +
                    rew_action_diff
                    )

    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.rpm[nth_drone, :]])
        return state.reshape(20,)
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)

        if np.abs(self.goal - state[0:3]).max() > MAX_Z:
            self.done_info = {'done': 'pos'}
            return True
        elif np.abs(state[7:9]).max() > np.pi / 2:
            self.done_info = {'done': 'roll_pitch'}
            return True
        elif np.linalg.norm(state[10:13]) > 10:
            self.done_info = {'done': 'lin_vel'}
            return True
        elif np.linalg.norm(state[13:16]) > np.pi:
            self.done_info = {'done': 'ang_vel'}
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.done_info = {'done': 'truncated'}
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        ## RECORD LAST STEP ACTION HERE SINCE THIS IS THE LAST LINE IN SELF.STEP
        info = {}
        info.update(self.rew_info)
        info.update(self.done_info)
        self.last_step_action = self.last_clipped_action[0]
        return info #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_rel_pos_z = np.clip(state[2] - self.goal[2], -MAX_Z, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_rel_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy # / MAX_XY not clipping xy since we only consider relative pos.
        normalized_pos_z = clipped_rel_pos_z # / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] # /np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_rpm = state[16:20] / self.MAX_RPM
        last_action = self.last_step_action

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      last_action
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1+RPM_FACTOR * action))
        elif self.ACT_TYPE == ActionType.TRPY:
            t, r, p, y = action
            m1 = t - r / 2 + p / 2 + y
            m2 = t - r / 2 - p / 2 - y
            m3 = t + r / 2 - p / 2 + y
            m4 = t + r / 2 + p / 2 - y
            rpm_normalized =  np.array([m1, m2, m3, m4])
            return np.array(self.HOVER_RPM * (1+0.5*rpm_normalized))
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(0)
            next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=self.goal,
                    step_size=1,
                )
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=next_pos
                                                 )
            rpm_input = rpm
        else:
            # print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")
            raise NotImplementedError("[ERROR] in BaseSingleAgentAviary._preprocessAction()")
        self.rpm = (1 / self.CTRL_FREQ / self.time_constant) * (rpm_input[np.newaxis, :] - self.rpm) + self.rpm
        self.control_hist.append(rpm_input / self.MAX_RPM)
        return self.rpm

    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        rpm = self.rpm[nth_drone, :] ## Use the dampled rpm
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        np.random.seed(seed)
        self.INIT_XYZS = (np.random.normal(0., 0.02, [3,]) + self.goal)[np.newaxis, :]
        self.INIT_RPYS = np.random.normal(0., 5./180.*np.pi, [3])[np.newaxis, :]

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### RESET THE RPM #########################################
        self._reset_rpm(seed)
        self._reset_hist()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()

        return initial_obs, initial_info



def plot_step_input():
    env = HoverAviaryDelay(gui=False,
                           act=ActionType.RPM
                           )
    env.reset()
    env.EPISODE_LEN_SEC = 3
    rpm = []
    for i in range(3 * env.CTRL_FREQ):
        action = np.ones(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        rpm.append(env.rpm)
        # env.render()
        # print(terminated)
        if terminated:
            obs, _ = env.reset(seed=42, options={})
    env.close()

    import matplotlib.pyplot as plt
    plt.plot(np.arange(3 * env.CTRL_FREQ), np.concatenate(rpm))
    plt.show()


if __name__ == '__main__':
    plot_step_input()

