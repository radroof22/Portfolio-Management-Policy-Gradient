version = "V1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

episode_reward_decay = .95
state_size = 5
action_size = 3
save_model_path = "E:\Documents\Code\Python\Data Science\Finance\Stock Market Bot\Saved_Models\V1_5LayerTorch\checkpoint_{}.model".format(version)
learning_rate = 1e-4
gamma = .95 # Decay rate of rewards
resume = True
action_mapper = {
    0: 'hold',
    1: 'buy',
    2: 'sell'
}

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 64)
        self.layer4 = nn.Linear(64, 16)
        self.layer5 = nn.Linear(16, action_size)

        self.action_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        action_score = F.relu(self.layer5(x))
        return F.softmax(action_score, dim=1)

def select_action(state):
    # Reformat State shape
    state = torch.from_numpy(state.values).float().unsqueeze(0).cuda()
    probs = policy.forward(state) # Run Forward prop
    m = Categorical(probs) # Create normalized distribution
    action = m.sample() # Choose action
    policy.action_probs.append(m.log_prob(action)) # Add distribution to stored for episode
    return action.item()

def finish_episode():
    R = 0 # Episode reward holder
    policy_loss = []
    
    # Deprecation of rewarsd over time
    rewards = []
    for r in policy.rewards[::-1]:
        R= r + gamma * R
        rewards.insert(0, R)
    
    rewards = torch.tensor(rewards).cuda()
    # Normalization of rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    # Compute loss
    for action_prob, reward in zip(policy.action_probs, torch.cuda.FloatTensor(rewards)):
        policy_loss.append(-action_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.action_probs[:]

def save_model():
    torch.save(policy.state_dict(), save_model_path)

def generate_action_call(action_key, amount=10):
    action_val = action_mapper[action_key]
    action = {
        "buy": 0,
        "sell": 0
    }
    if action != "hold":
        action[action_val] = amount
    return action


policy = Policy().cuda()

if resume:
    policy.load_state_dict(torch.load(save_model_path))

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
