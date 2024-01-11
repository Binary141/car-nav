Uniform Coins Environment
-------------------------

# Description

The state is a collection of `N` coins that are either 
heads or tails. The goal is to turn over coins until 
all coins are the same.

# Observation Space

An observation is a numpy.array of `int8`, and shape = (N,).
Each entry is either `0` or `1`, representing heads or tails
respectively.

# Action Space

An action is an integer representing the index in the state
of which coin to turn over.

# Starting State

The starting state is a randomly selected state for each of the `N` coins.

# Rewards

The base cost of turning a coin is 1 unit. However,
the heads side of a coin has slightly more mass than 
the tails side. So turning the coin from heads to tails
recovers a little potential energy. Turning from tails to
heads requires storing a small amount of energy.

Turning from heads to tails has a total reward of -0.9.
Turning from tails to heads has a total reward of -1.1.

# Episode End

The episode terminates when all of the coins have the same
side exposed. Either all heads or all tails. 

The episode will truncate after 50 steps. This limit can
be overridden with the `max_episode_steps` parameter to 
`gymnasium.make()`.

