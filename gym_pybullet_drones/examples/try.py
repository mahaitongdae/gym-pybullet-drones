import gymnasium as gym
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import time

env = HoverAviary(gui=True,
                      record=False,
                      act= ActionType.RPM,
                  initial_rpys=np.array([0,0,0])[np.newaxis, :]
                     )
logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                num_drones=1,
                # output_folder=output_folder,
                # colab=colab
                )
obs, info = env.reset(seed=42, options={})
start = time.time()
for i in range(3*env.CTRL_FREQ):
    action = np.ones([4,])
    # action = env.goal
    obs, reward, terminated, truncated, info = env.step(action)
    logger.log(drone=0,
               timestamp=i/env.CTRL_FREQ,
               state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
               control=np.zeros(12)
               )
    env.render()
    # time.sleep(0.1)
    print(terminated)
    sync(i, start, env.CTRL_TIMESTEP)
    if terminated:
        obs, _ = env.reset(seed=42, options={})
env.close()

# if plot:
logger.plot()