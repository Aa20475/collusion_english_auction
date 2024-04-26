# This demo script will load the saved policy model and run an episode of the auction


import os
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from envs.english_auction_env import EnglishAuctionEnv
from train import build_input_from_obs_info

number_of_agents = 2
number_of_items = 3
budget_limit = 10

# initialize the environment
env = EnglishAuctionEnv(num_agents=number_of_agents, num_objects=number_of_items, budget_limit=budget_limit, no_bid_rounds=3)

# load the saved policy model
model_path = "./2agents_3obj_3nbr_10budget/policy_net_2agents_3obj_3nbr_10budget.th"
policy_net = torch.load(model_path)


# run an episode of the auction
env.demo = True
obs, info= env.reset()
state = build_input_from_obs_info(obs, info)
done = False

while not done:
    with torch.no_grad():
        action = policy_net(state).max(1).indices
    obs, reward, terminated, truncated, info = env.step(action)
    state = build_input_from_obs_info(obs, info)
    done = terminated or truncated

    
