#!/usr/bin/env python3
import gymnasium as gym
import car_navigation
import random
import queue
import time
import argparse
import os
import sys

import bfs
import dfs
import astar
import greedy

def main(iterations=10, row_count=5, column_count=5, searches=[], debug=False):
    model = car_navigation.CarNavigationModel

    render_mode = "ansi"

    for search in searches:
        averageNodes = 0
        nodesGenerated = []

        averageTime = 0
        times = []

        for i in range(iterations):
            print(f"Working on iteration {i}")
            env = gym.make('car_navigation/CarNavigation-v0', render_mode=render_mode, row_count=row_count, column_count=column_count)
            observation, info = env.reset()
            state = car_navigation.CarNavigationState(row_count=row_count, column_count=column_count)
            state.observation = observation
            if debug:
                print(state)

            terminated = truncated = False

            startTime = time.time()
            steps, newState, total_states_generated = search(state, model)
            endTime = time.time()
            if debug:
                print(steps)

            averageTime += (endTime - startTime)
            times.append((endTime - startTime))

            averageNodes += total_states_generated
            nodesGenerated.append(total_states_generated)

            env.close()

        nodesGenerated.sort()
        times.sort()
        output = f"============== {search.__name__} rows: {row_count} cols: {column_count} ==============\n"
        output += f"average nodes generated: {averageNodes / iterations}\n"
        output += f"Median nodes generated: {nodesGenerated[len(nodesGenerated) // 2]}\n"
        output += f"All node counts list: {nodesGenerated}\n"
        output += "\n"
        output += f"averageTime: {averageTime / iterations}\n"
        output += f"times: {times[len(times) // 2]}\n"
        output += f"times list: {times}\n"
        output += f"======================================================================================\n\n"

        f = open(f"{search.__name__}_{row_count}_{column_count}.txt", "a")
        f.write(output)
        f.close()

        print(output)
    return

if __name__ == "__main__":
    sys.setrecursionlimit(6000)
    print(f"PID: {os.getpid()}")
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--i', type=int, help='how many iterations to do')
    parser.add_argument('--r', type=int, help='how many rows to use')
    parser.add_argument('--c', type=int, help='how many columns to use')
    parser.add_argument('--astar', action='store_true', help='enable ASTAR search')
    parser.add_argument('--greedy', action='store_true', help='enable GREEDY search')
    parser.add_argument('--bfs', action='store_true', help='enable BFS search')
    parser.add_argument('--dls', action='store_true', help='enable DLS search')
    parser.add_argument('--debug', action='store_true', help='enable debug')

    args = parser.parse_args()
    print(args)
    print()

    if args.i == None:
        args.i = 10
    if args.r == None:
        args.r = 5
    if args.c == None:
        args.c = 5

    searches = []
    if args.greedy:
        searches.append(greedy.GREEDY)
    if args.astar:
        searches.append(astar.ASTAR)
    if args.bfs:
        searches.append(bfs.BFS)
    if args.dfs:
        searches.append(dfs.DLS)

    main(iterations=args.i, row_count=args.r, column_count=args.c, searches=searches, debug=args.debug)
