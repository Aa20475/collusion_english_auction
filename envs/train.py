import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from itertools import count
import math
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from english_auction_env import EnglishAuctionEnv

env = EnglishAuctionEnv(num_agents=2, num_objects=3, no_bid_rounds=3, budget_limit=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    # n_observations: auction State + flattened_action_history
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 2
# Get the number of state observations
state, info = env.reset()


# observations for each agent
n_observations = 0
n_observations += 3 # agent_prefs
n_observations += env.num_objects # current_object one-hot
n_observations += env.num_objects # objects won so far
n_observations += 3 * env.num_objects # all object_stats
n_observations += 1 # agent_budget 
n_observations += 1 # current_bidding_price
n_observations += 1 # is current agent the highest bidder
n_observations += 1 # bid_increment
n_observations += 1 # number of no_bid_rounds right now
n_observations += env.num_agents * env.bid_history_size # flattened_action_history


def build_input_from_obs_info(obs, info):
    # Initialize a list to hold the input tensors for each agent
    input_tensors = []

    # Loop over each agent
    for i in range(len(info['agent_prefs'])):
        # Extracting the observation and info for each agent
        agent_prefs = torch.tensor([info['agent_prefs'][i]], dtype=torch.float32, device=device)
        current_object = torch.tensor([obs['object_id']], dtype=torch.float32, device=device)
        objects_won_so_far = torch.zeros((1,env.num_objects), dtype=torch.float32, device=device)
        for object_won in info['agent_objects_won'][i]:
            objects_won_so_far[0][object_won] = 1
        all_object_stats = torch.tensor([obs['stats']], dtype=torch.float32, device=device)
        agent_budget = torch.tensor([[info['remaining_budgets'][i]]], dtype=torch.float32, device=device)
        current_bidding_price = torch.tensor([obs['current_bid']], dtype=torch.float32, device=device)
        is_current_agent_highest_bidder = torch.tensor([obs['current_bid'] == info['remaining_budgets'][i]], dtype=torch.float32, device=device)
        bid_increment = torch.tensor([obs['bid_increment']], dtype=torch.float32, device=device)
        number_of_no_bid_rounds = torch.tensor([obs['no_bid_rounds']], dtype=torch.float32, device=device)
        flattened_action_history = torch.tensor([obs['action_history']], dtype=torch.float32, device=device)

        # Concatenating all the tensors to form a single input tensor for each agent
        input_tensor = torch.cat((agent_prefs, current_object, objects_won_so_far, all_object_stats, agent_budget, current_bidding_price, is_current_agent_highest_bidder, bid_increment, number_of_no_bid_rounds, flattened_action_history), dim=1)

        # Append the input tensor to the list
        input_tensors.append(input_tensor)

    # Stack the input tensors to get a tensor of shape (num_agents, n_observations)
    input_tensors = torch.cat(input_tensors, dim=0)

    return input_tensors

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

obs, info = env.reset()
state = build_input_from_obs_info(obs, info)

def select_action(state):
    # per agent action
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices
    else:
        # random 0 or 1
        return torch.randint(2, (env.num_agents, ), device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(-1, action_batch.unsqueeze(-1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(action_batch.shape, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

episode_durations = []

def plot_stats(cumulative_reward_per_episode, reward_tracking_window, average_winning_bid_per_episode, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if not show_result:
        plt.clf()
    
    plt.title("Auction Length")
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    if show_result:
        plt.show()

    # Plotting Average Reward per Episode: Sum of rewards obtained by an agent within an episode, averaged over a window of episodes
    # cumulative_reward_per_episode: shape (num_episodes, num_agents) containing the cumulative reward obtained by each agent in each episode
    # plot one line for the average cumulative reward showing the average reward obtained by all agents over a window of episodes
    average_cumulative_reward_per_episode = torch.mean(torch.stack(cumulative_reward_per_episode), dim=1).to(device)
    if len(average_cumulative_reward_per_episode) >= reward_tracking_window:
        moving_average_cumulative_reward = average_cumulative_reward_per_episode.unfold(0, reward_tracking_window, 1).mean(1).view(-1).to(device)
        moving_average_cumulative_reward = torch.cat((torch.zeros(reward_tracking_window-1, device=device), moving_average_cumulative_reward))
    elif len(average_cumulative_reward_per_episode) > 0:
        moving_average_cumulative_reward = torch.full((len(average_cumulative_reward_per_episode),), average_cumulative_reward_per_episode.mean(), device=device)
    else:
        moving_average_cumulative_reward = torch.tensor([], device=device)
    plt.figure(2)
    if not show_result:
        plt.clf()
    
    plt.title("Average Episode Cumulative Reward")
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')
    plt.plot(moving_average_cumulative_reward.cpu().numpy())
    if show_result:
        plt.show()

    # we plot the average winning bid per episode
    plt.figure(3)
    if not show_result:
        plt.clf()
    plt.title("Average Winning Bid")
    plt.xlabel('Episode')
    plt.ylabel('Average Winning Bid')
    plt.plot(average_winning_bid_per_episode)
    if show_result:
        plt.show()

    plt.pause(0.001)  # pause a bit so that plots are updated

if torch.cuda.is_available():
    num_episodes = 2000
    reward_tracking_window = 10
else:
    num_episodes = 100
    reward_tracking_window = 2


cumulative_reward_per_episode = []
average_winning_bid_per_episode = []
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    obs, info = env.reset()
    state = build_input_from_obs_info(obs, info)

    # Initialize the cumulative reward for the episode
    cumulative_reward = torch.zeros(env.num_agents, device=device)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor(reward, device=device)
        cumulative_reward += reward
        reward = reward.unsqueeze(0)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = build_input_from_obs_info(observation, info).to(device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state.squeeze(0) if next_state is not None else None

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            cumulative_reward_per_episode.append(cumulative_reward)
            # compute average winning bid from the current_bid of each object
            average_winning_bid = sum([obs.current_bid for obs in env.objects])/env.num_objects
            average_winning_bid_per_episode.append(average_winning_bid)

            plot_stats(cumulative_reward_per_episode, reward_tracking_window, average_winning_bid_per_episode)
            break

plot_stats(cumulative_reward_per_episode, reward_tracking_window, average_winning_bid_per_episode, show_result=True)
print('Complete')