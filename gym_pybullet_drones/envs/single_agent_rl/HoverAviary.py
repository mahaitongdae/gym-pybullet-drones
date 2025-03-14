import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, \
    BaseSingleAgentAviary
from gymnasium import spaces
from collections import deque
from copy import deepcopy

MAX_LIN_VEL_XY = 3
MAX_LIN_VEL_Z = 1
MAX_ANG_VEL = 2.
MAX_XY = 1.
MAX_Z = 1.
MAX_PITCH_ROLL = np.pi  # Full range
MAX_INT_POS_ERROR = 1
MAX_DIFF_POS_ERROR_XY = 2
MAX_DIFF_POS_ERROR_Z = 0.15
MAX_INT_RPY_ERROR = 1
MAX_DIFF_RPY_ERROR = 1
RPM_FACTOR = 0.2
UINT16_MAX = 65535

state_low = [-MAX_XY, -MAX_XY, 0.0,  # XYZ
                   0.0, 0.0, 0.0, 0.0,  # Quat
                   - MAX_PITCH_ROLL, -MAX_PITCH_ROLL, -np.pi,  # RPY
                   -MAX_LIN_VEL_XY, -MAX_LIN_VEL_XY, -MAX_LIN_VEL_Z,  #Lin vel
                   -MAX_ANG_VEL, -MAX_ANG_VEL, -MAX_ANG_VEL,  # ANG_VEL
                   ]

action_low = [-1., -1., -1., -1.,]

int_dif_low = [-MAX_INT_POS_ERROR, -MAX_INT_POS_ERROR, -MAX_INT_POS_ERROR, #
                   -MAX_DIFF_POS_ERROR_XY, -MAX_DIFF_POS_ERROR_XY, -MAX_DIFF_POS_ERROR_Z,
                   -MAX_INT_RPY_ERROR, -MAX_INT_RPY_ERROR, -MAX_INT_RPY_ERROR,
                   -MAX_DIFF_RPY_ERROR, -MAX_DIFF_RPY_ERROR, -MAX_DIFF_RPY_ERROR,]

state_high = [MAX_XY, MAX_XY, 2.0,
                    1.0, 1.0, 1.0, 1.0,
                    MAX_PITCH_ROLL, MAX_PITCH_ROLL, np.pi,
                    MAX_LIN_VEL_XY, MAX_LIN_VEL_XY, MAX_LIN_VEL_Z,
                    MAX_ANG_VEL, MAX_ANG_VEL, MAX_ANG_VEL,
                    ]

action_high = [1., 1., 1., 1.,]

int_dif_high = [ MAX_INT_POS_ERROR, MAX_INT_POS_ERROR, MAX_INT_POS_ERROR,
                   MAX_DIFF_POS_ERROR_XY, MAX_DIFF_POS_ERROR_XY, MAX_DIFF_POS_ERROR_Z,
                   MAX_INT_RPY_ERROR, MAX_INT_RPY_ERROR, MAX_INT_RPY_ERROR,
                   MAX_DIFF_RPY_ERROR, MAX_DIFF_RPY_ERROR, MAX_DIFF_RPY_ERROR,]


class HoverAviary(BaseSingleAgentAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PWM,
                 add_action_obs=True,
                 add_pd=False
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
            initial_xyzs = np.random.normal(0., 0.02, [3, ])
            initial_xyzs = self.goal + initial_xyzs
            initial_xyzs = initial_xyzs[np.newaxis, :]
            self.INIT_XYZS = None
        else:
            self.INIT_XYZS = initial_xyzs
        if initial_rpys is None:
            initial_rp = np.random.normal(0., 5. / 180. * np.pi, [2, ])
            initial_y = np.random.normal(0., 60. / 180. * np.pi, )
            initial_rpys = np.hstack([initial_rp, initial_y])
            initial_rpys = initial_rpys[np.newaxis, :]
            self.INIT_RPYS = None
        else:
            self.INIT_RPYS = initial_rpys

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
        self.EPISODE_LEN_SEC = 4
        self.rew_info = {}
        self.done_info = {}
        self.last_step_action = np.zeros([4, ])
        self.reset_errors()

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.HOVER_PWM = (self.HOVER_RPM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE / 65535 * 2 - 1
        self.rew_buf = {'rew_pos': 0,
                         'rew_rpy': 0,
                         'rew_lin_vel': 0,
                         'rew_ang_vel': 0,
                         'rew_action': 0,
                         'rew_action_diff': 0}

    def reset_errors(self):
        # self.pos_hist = deque(maxlen=hist_horizon)
        # for _ in range(hist_horizon):
        #     self.pos_hist.append(np.zeros([3]))
        # self.ang_hist = deque(maxlen=hist_horizon)
        # for _ in range(hist_horizon):
        #     self.ang_hist.append(np.zeros([3]))
        # self.control_hist = deque(maxlen=hist_horizon)
        # for _ in range(hist_horizon):
        #     self.control_hist.append(np.zeros([4]))
        # self.add_hist()
        self.last_pos = deepcopy(self.pos[0])
        self.intergral_pos_e = np.zeros([3, ])  # self.goal - self.pos[0]
        self.last_rpy = deepcopy(self.rpy[0])
        self.intergral_rpy_e = np.zeros([3, ])  # - self.rpy[0]

    ################################################################################

    def update_int_errors(self):
        self.intergral_pos_e += (self.goal - self.pos[0]) / self.CTRL_FREQ
        self.intergral_rpy_e += (np.zeros([3, ]) + self.rpy[0]) / self.CTRL_FREQ

    def update_diff_errors(self):
        self.last_pos = deepcopy(self.pos[0])
        self.last_rpy = deepcopy(self.rpy[0])

    def get_pos_error(self):
        pos_error_d = (self.pos[0] - self.last_pos) * self.CTRL_FREQ  # actually vel error
        pos_error_i = self.intergral_pos_e
        pos_error_i = np.clip(pos_error_i, -2, 2)
        pos_error_i[2] = np.clip(pos_error_i[2], -0.15, 0.15)
        return pos_error_i, pos_error_d

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        state[8] = -1 * state[8]
        state[14] = -1 * state[14]
        # obs = self._clipAndNormalizeState(state)
        obs = state[:16]
        ############################################################
        #### OBS OF SIZE 16 (WITH QUATERNION AND RPMS)
        # return obs
        ############################################################
        #### OBS SPACE OF SIZE 16 xyz 3, quat 4, rpy 3, vel_xyz 3, angle_vel_xyz 3 each
        # ret = np.hstack([obs[0:10], obs[10:13], obs[13:16]]).reshape(12, )
        ret = obs
        if self.add_action_obs:
            ret = np.hstack([ret, np.squeeze(self.last_action)])
        if self.add_pd:
            self.update_int_errors()
            # pos_error_i = -1 * np.array(self.pos_hist).sum(axis=0) / self.CTRL_FREQ
            pos_error_i, pos_error_d = self.get_pos_error()
            ang_error_d = (self.rpy[0] - self.last_rpy) * self.CTRL_FREQ
            ang_error_i = self.intergral_rpy_e
            ang_error_i = np.clip(ang_error_i, -np.pi, np.pi)
            ang_error_i[0:2] = np.clip(ang_error_i[0:2], -1, 1)
            self.update_diff_errors()
            return np.hstack([ret, pos_error_i, pos_error_d, ang_error_i, ang_error_d]).astype('float32')
        else:
            return ret.astype('float32')

    def _observationSpace(self):
        state_space_low = state_low
        state_space_high = state_high
        if self.add_action_obs:
            state_space_low += action_low
            state_space_high += action_high
        if self.add_pd:
            state_space_low += int_dif_low
            state_space_high += int_dif_high
        return spaces.Box(low=np.array(state_space_low),
                          high=np.array(state_space_high),
                          dtype=np.float32
                          )

    def _actionSpace(self):
        return spaces.Box(low=-1 * np.ones([4]),
                          high=np.ones([4]),
                          dtype=np.float32)

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
        rew_pos = - 2.5 * np.linalg.norm(self.goal - state[0:3])
        rew_rpy = - 0.1 * np.linalg.norm(state[7:10])
        rew_lin_vel = - 0.05 * np.linalg.norm(state[10:13])
        rew_ang_vel = - 0.05 * np.linalg.norm(state[13:16])
        rew_action = - 0.1 * np.linalg.norm(self.raw_action[0] - self.HOVER_PWM) # self.last_clipped_action[0] / self.MAX_RPM
        rew_action_diff = -0.1 * np.linalg.norm(
            (self.raw_action[0] - self.last_action[0]) ) # / (2 * RPM_FACTOR * self.HOVER_RPM)
        self.rew_info = {'rew_pos': rew_pos,
                         'rew_rpy': rew_rpy,
                         'rew_lin_vel': rew_lin_vel,
                         'rew_ang_vel': rew_ang_vel,
                         'rew_action': rew_action,
                         'rew_action_diff': rew_action_diff}
        for key, value in self.rew_buf.items():
            self.rew_buf[key] = value + self.rew_info[key]
        return np.exp(rew_pos +
                    rew_rpy +
                    rew_lin_vel +
                    rew_ang_vel +
                    rew_action +
                    rew_action_diff
                    )

    ################################################################################

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
        # elif np.linalg.norm(state[13:16]) > np.pi:
        #     self.done_info = {'done': 'ang_vel'}
        #     return True
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
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
        info.update({'instant_rew': self.rew_info})
        info.update({'episode_rew': self.rew_buf})
        info.update(self.done_info)
        self.last_step_action = self.last_clipped_action[0]
        return info  #### Calculated by the Deep Thought supercomputer in 7.5M years

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

        # normalized_pos_xy = clipped_pos_xy # / MAX_XY not clipping xy since we only consider relative pos.
        # normalized_pos_z = clipped_rel_pos_z # / MAX_Z
        # normalized_rp = clipped_rp / MAX_PITCH_ROLL
        # normalized_y = state[9] / np.pi # No reason to clip
        # normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        # normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        # normalized_ang_vel = state[13:16] # /np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        # normalized_rpm = state[16:20] / self.MAX_RPM
        # last_action = self.last_step_action
        state_buf = [clipped_pos_xy,  # x, y
                      clipped_rel_pos_z,  # z_error
                      state[3:7],  # quat
                      clipped_rp,  # row, pitch
                      state[9],  # yaw
                      clipped_vel_xy,  # vel xy
                      clipped_vel_z,  # vel z
                      state[13:16],  # angular vel
                      # last_action
                        ]
        if self.add_action_obs:
            state_buf.append(self.last_action[0])

        norm_and_clipped = np.hstack(state_buf)

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
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                        state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                       state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                        state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

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
        # assert self.ACT_TYPE == ActionType.TRPY
        if self.ACT_TYPE == ActionType.RPM:
            return self.HOVER_RPM + action * RPM_FACTOR * self.MAX_RPM
        elif self.ACT_TYPE == ActionType.PWM:
            pwm = UINT16_MAX * (action + 1) / 2
            rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
            return rpm
        elif self.ACT_TYPE == ActionType.RAW:
            # RAW for force input
            return np.array(np.sqrt(0.25 * self.MAX_THRUST * (0.5 * (1 + action)) / self.KF))
        elif self.ACT_TYPE == ActionType.TRPY:
            t, r, p, y = action
            thrust = self.MAX_THRUST + t * (self.MAX_THRUST - self.GRAVITY) / 4
            mr = r * self.MAX_XY_TORQUE
            mp = p * self.MAX_XY_TORQUE
            my = y * self.MAX_Z_TORQUE
            fr = mr / (self.L / np.sqrt(2))
            fp = mp / (self.L / np.sqrt(2))
            fy = my / self.KM * self.KF
            f1 = t - r / 2 + p / 2 + y
            f2 = t - r / 2 - p / 2 - y
            f3 = t + r / 2 - p / 2 + y
            f4 = t + r / 2 + p / 2 - y
            force = np.array([f1, f2, f3, f4]).clip(0.0, self.MAX_THRUST / 4)
            rpm = np.sqrt(force / self.KF)
            return rpm
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
            return rpm
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")

    def reset(self,
              seed: int = None,
              options: dict = None):
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
        if self.INIT_XYZS is None:
            self.INIT_XYZS = (np.random.normal(0., 0.02, [3, ]) + self.goal)[np.newaxis, :]
        if self.INIT_RPYS is None:
            self.INIT_RPYS = np.random.normal(0., 5. / 180. * np.pi, [3])[np.newaxis, :]
        # self.INIT_XYZS = np.zeros([3,])[np.newaxis, :]
        # self.INIT_RPYS = np.zeros([3,])[np.newaxis, :]

        p.resetSimulation(physicsClientId=self.CLIENT)

        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        self.reset_errors()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()

        for key, _ in self.rew_buf.items():
             self.rew_buf[key] = 0
        return initial_obs, initial_info


def test_custom_drone_env():
    from envs.HoverAviary import HoverAviary
    import gymnasium as gym
    env = gym.make('hover-aviary-v0')
    env.reset()
    action = env.action_space.sample()
    print(env.step(action))

