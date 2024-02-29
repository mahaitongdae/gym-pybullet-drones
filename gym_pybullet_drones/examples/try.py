import gymnasium as gym
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

env = gym.make("hover-aviary-v0")
env.reset()
env.step(env.action_space.sample())