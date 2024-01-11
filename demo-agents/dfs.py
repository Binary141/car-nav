#!/usr/bin/env python3
import gymnasium as gym
import car_navigation
import random

def DLS(state, model, max_depth=50):
    Q = []
    # state, action, parent, depth
    S = (state, None, None, 0)
    Q.append(S)
    total_states_generated = 0
    return_steps = []

    while len(Q) > 0:
        S1 = Q.pop()
        temp_state, action, parent, depth = S1

        if model.GOAL_TEST(temp_state):
            while parent is not None:
                return_steps.append(action)
                temp_state, action, parent, depth = parent
            return_steps.reverse()
            return True, return_steps, total_states_generated

        if (depth + 1)< max_depth:
            for action in model.ACTIONS(temp_state):
                total_states_generated += 1
                # Got here from S using action
                S2 = (model.RESULT(temp_state, action),) + (action, S1, depth + 1)
                # Insert as first item in the list
                Q.insert(0, S2)
    return False, return_steps, total_states_generated

def IDS(state, model):
    max_iters = 50
    for max_depth in range(max_iters):
        found, path, total_states_generated = DLS(state, model, max_depth)
        if found:
            return path, total_states_generated
        else:
            continue
    raise Exception("Couldn't find goal with a max depth of ", max_iters)

def main():
    model = car_navigation.CarNavigationModel

    render_mode = "ansi"

    row_count = 5
    column_count = 5
    env = gym.make('car_navigation/CarNavigation-v0', render_mode=render_mode, row_count=row_count, column_count=column_count)
    observation, info = env.reset()
    state = car_navigation.CarNavigationState(row_count=row_count, column_count=column_count)
    state.observation = observation

    terminated = truncated = False

    print(state)
    steps, total_states_generated = IDS(state, model)
    print(f"{steps}\n")
    print(total_states_generated)

    env.close()
    return

if __name__ == "__main__":
    main()
