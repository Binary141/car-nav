import gymnasium
import numpy as np
from gymnasium import spaces
from car_navigation.envs.car_navigation_model import CarNavigationModel
from car_navigation.envs.car_navigation_model import CarNavigationState

try:
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, `pip install` must have failed."
    ) from e

class CarNavigationEnv(gymnasium.Env):

    metadata = {
        "render_modes": ["rgb_array", "ansi"],
        "render_fps": 1,
    }

    def __init__(self, render_mode=None, row_count=1, column_count=5, num_flipped = 2):
        self.render_mode = render_mode
        self.row_count = row_count
        self.column_count = column_count
        self.num_flipped = num_flipped
        self.action_space = spaces.Discrete(row_count * column_count)
        self.observation_space = spaces.Box(0, 3, shape=(row_count,column_count), dtype=np.int8)

        self.window_surface = None
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = CarNavigationState(self.row_count, self.column_count, self.num_flipped)
        self.state.randomize(seed)

        observation = self.state.observation
        info = {}
        return observation, info

    def step(self, action):
        state = self.state
        state1 = CarNavigationModel.RESULT(state, action)
        self.state = state1

        observation = self.state.observation
        reward = CarNavigationModel.STEP_COST(state, action, state1)
        terminated = CarNavigationModel.GOAL_TEST(state1)
        info = {}

        # display support
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "human":
            return self._render_text()

    def _render_text(self):
        return str(self.state)

    def _render_gui(self, mode):
        return str(self.state)

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
        return
