LOAD = True
OUTPUT_PATH = "models/regular/model_001.pt"
eps=1e-10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

from Environment2 import Environment
env = Environment()

hparams = {
    "learning_rate": 1e-3,
    "gamma": 0.99
}

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.action_space = env.action_space
        self.state_space = env.observation_space

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.saved_log_probs = []


    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

def select_action(state):
    state = torch.from_numpy(state).float().cuda().unsqueeze(0)
    probs = agent(state)
    m = Categorical(probs)
    action = m.sample()
    agent.saved_log_probs.append(m.log_prob(action))
    return probs

def format_action(action_in:list):
    action = {
        "hold": 0,
        "buy": 0,
        "sell": 0
    }
    
    # [hold, buy, sell]
    max_action = torch.argmax(action_in, dim=-1)
    
    if max_action == 1:
        action["buy"] = 1
    elif max_action == 2:
        action["sell"] = 1
    else:
        action["hold"] = 1

    return action

def update_agent():
    R = 0
    policy_loss = []
    rewards = []
    for r in agent.ep_rewards[::-1]:
        R = r + hparams["gamma"] * R
        rewards.insert(0, R)
    
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards).cuda()
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

    # TODO: LOGPROBS
    
    for log_prob, reward in zip(agent.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    # optimize gradient
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del agent.saved_log_probs[:]

agent = Agent().cuda()
if LOAD:
    agent.load_state_dict(torch.load(OUTPUT_PATH))
optimizer = optim.Adam(agent.parameters(), lr=hparams["learning_rate"])

if __name__ == "__main__":
    
    # running_reward = 10
    for episode in range(1000):
        state = env.reset() # Reset environment and record the starting state
        
        # Reset episode records
        agent.ep_states = []
        agent.ep_actions = []
        agent.ep_rewards = []

        time = 0
        running_reward = 10
        while True:
            
            # Step through environment using chosen action
            actions = select_action(state.values.reshape(1, -1))
            action_dict = format_action(actions)

            state, reward, done = env.step(action_dict)

            # Record Action and Reward reference to State
            agent.ep_rewards.append(reward)
            
            # Check if done
            if done:
                break
            time += 1

        update_agent()
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        

        if episode % 50 == 0:
                    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
                    torch.save(agent.state_dict(), OUTPUT_PATH)
        if running_reward > 10000:
                    print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
                    break


