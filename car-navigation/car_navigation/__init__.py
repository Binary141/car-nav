from gymnasium.envs.registration import register

from car_navigation.envs.car_navigation_env import CarNavigationEnv
from car_navigation.envs.car_navigation_model import CarNavigationModel
from car_navigation.envs.car_navigation_model import CarNavigationState

register(
    # car_navigation is this folder name
    # -v0 is because this first version
    # CarNavigation is the pretty name for gym.make
    id="car_navigation/CarNavigation-v0",

    # car_navigation.envs is the path car_navigation/envs
    # CarNavigationEnv is the class name
    entry_point="car_navigation.envs:CarNavigationEnv",

    # configure the automatic wrapper to truncate after 50 steps
    max_episode_steps=50,
)
