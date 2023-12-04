#!/usr/bin/env python3

import gymnasium as gym
import uniform_coins
import random

def agent_function(state):
    """
    state: A uniform_coins.UniformCoinsState object. The current state of the environment.
    
    returns: An integer, the coin to turn over.
    """
    action = random.choice(uniform_coins.UniformCoinsModel.ACTIONS(state))
    return action

def main():
    coin_count = 9

    # render_mode = None
    # render_mode = "ansi"
    # render_mode = "rgb_array"
    render_mode = "human"

    env = gym.make('uniform_coins/UniformCoins-v0', render_mode=render_mode, coin_count=coin_count)
    observation, info = env.reset()
    state = uniform_coins.UniformCoinsState()
    state.observation = observation
    
    terminated = truncated = False
    if render_mode == "ansi":
        print("Current state:", env.render())
    while not (terminated or truncated):
        action = agent_function(state)
        if render_mode == "ansi":
            print()
            print(f"Action: turn coin {action}.")
        observation, reward, terminated, truncated, info = env.step(action)
        state.observation = observation
        if render_mode == "ansi":
            print("Current state:", env.render())

    env.close()
    return

if __name__ == "__main__":
    main()
    

