#!/usr/bin/env python3
import gymnasium as gym
import car_navigation
import random

def BFS(state, model):
    Q = []
    # state, action, parent
    S = (state, None, None) #.getAgentLocation() + (None, None, 0)
    Q.append(S)
    total_states_generated = 0

    while len(Q) > 0:
        S1 = Q.pop(0)
        # col, row, action, parent, depth = S1
        temp_state, action, parent = S1
        if model.GOAL_TEST(temp_state):
            steps = []
            while parent is not None:
                steps.append(action)
                state, action, parent = parent
                # col, row, action, parent, depth = parent

            steps.reverse()
            return steps, temp_state, total_states_generated

        for action in model.ACTIONS(temp_state):
            total_states_generated += 1
            # Got here from S using action
            S2 = (model.RESULT(temp_state, action),) + (action, S1)
            Q.append(S2)

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
    steps, newState, total_states_generated = BFS(state, model)
    print(steps)
    print()
    print(newState)
    print(total_states_generated)

    env.close()
    return

if __name__ == "__main__":
    main()
