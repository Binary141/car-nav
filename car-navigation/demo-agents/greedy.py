#!/usr/bin/env python3
import gymnasium as gym
import car_navigation
import random
import queue

# @profile
def GREEDY(state, model):
    reached = {}
    Q = queue.PriorityQueue()
    Q.put(state)
    total_states_generated = 1

    # key = tuple(state.getAgentLocation()) + state.observation.data
    agent_row, agent_col = state.getAgentLocation()
    hashPaths = hash(str(state._paths))

    key = (agent_row, agent_col, hashPaths)
    key = tuple(state.getAgentLocation()) + tuple(map(tuple, state.observation.tolist()))
    reached[key] = state
    bestHeuristic = 1_000_000

    while not Q.empty():
        S = Q.get()

        if model.GOAL_TEST(S):
            path = []
            temp = S
            while temp._parent is not None:
                path.append(temp._action)
                temp = temp._parent

            path.reverse()
            return list(path), None, total_states_generated

        for action in model.ACTIONS(S):
            total_states_generated += 1
            S1 = model.RESULT(S, action)
            heuristic = model.HEURISTIC(S1)
            S1._estimated_cost = heuristic

            if heuristic < bestHeuristic:
                # print(f"best heuristic so far: {heuristic}")
                bestHeuristic = heuristic
                # print(S1)

            agent_row, agent_col = S1.getAgentLocation()
            # print(agent_row, agent_col)
            hashPaths = hash(str(S1._paths))

            key = (agent_row, agent_col, hashPaths)
            key = tuple(S1.getAgentLocation()) + tuple(map(tuple, S1.observation.tolist()))
            #key = (agent_row, agent_col, hashPaths)
            if (key not in reached) or ((heuristic) < (reached[key]._estimated_cost)):
                S1._parent = S
                S1._action = action

                reached[key] = S1
                Q.put(S1)

    raise Exception("Couldn't find anything!")

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

    steps, newState, total_states_generated = GREEDY(state, model)

    print(f"Total states generated: {total_states_generated}")

    print(steps)
    for action in steps:
        state = model.RESULT(state, action)
        print(state)

    env.close()
    return

if __name__ == "__main__":
    main()
