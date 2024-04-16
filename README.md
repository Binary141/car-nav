# Uniform Coins

## Background
This is the final project of CS 4300 in which we extend a [Gymnasium](https://gymnasium.farama.org/)  environment to create our own problem and solution.
 This project sets up an NxN grid with valid tiles containing one of {/, \, |, -} where the agent can rotate the tile, or move in the direction of the current tile. For example, if the agent is on a tile with a '|', then it can move to, and rotate, the tiles above and below it's current position.

## Setup
First run `make pip-install` to install the dependencies for the project. Then run `make install-car-nav` to install the local Python package. That's it!

## Usage
Within the `demo-agents` folder there are some different searches and examples on how to use it. The `benchmark.py` file would be the best place to start by running `python3 benchmark.py -h` to see the possible options that can be run

![Example output](https://github.com/Binary141/car-nav/blob/main/sample_astar_output.png)
