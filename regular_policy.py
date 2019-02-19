LOAD = True
OUTPUT_PATH = "models/regular/model_proper_1000k.pt" # _500Cash
eps=1e-10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.autograd import Variable

from Environment import Environment
env = Environment()

hparams = {
    "learning_rate": 1e-6,
    "gamma": 0.99
}

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.action_space = env.action_space
        self.state_space = env.observation_space

        self.lstm = nn.LSTM(self.state_space[1], 64)
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 16)
        self.l5 = nn.Linear(16, self.action_space)

        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor().cuda())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
 
    def forward(self, x):
        for day in x[0]:
            out, state = self.lstm(day.view(1, 1, 5))

        out = F.dropout(out, p=0.6)

        out = self.l1(out)
        out = F.dropout(out, p=0.4)
        out = F.relu(out)
        
        out = self.l2(out)
        out = F.dropout(out, p=0.5)
        out = F.relu(out)
        
        out = self.l3(out)
        out = F.dropout(out, p=0.5)
        out = F.relu(out)
        
        out = self.l4(out)
        out = F.dropout(out, p=0.4)
        out = F.relu(out)
        
        out = self.l5(out)
        out = F.softmax(out, dim=-1)
        
        return out

def select_action(state):
    state = (state - state.mean()) / (state.max() - state.min())
    
    state = torch.from_numpy(state).float().cuda().unsqueeze(0)
    
    if torch.isnan(state[0][0][0]):
        print(agent.portfolio["balance"])
        return 0
    probs = agent(state)
    
    c = Categorical(state)
    action = c.sample()
    
    if torch.isnan(probs[0][0][0]):
        print(probs)
        print(list(agent.parameters()))
        print("*"*120)
        print(state)
        print("*"*120)
        print(env.portfolio["balance"])
        import sys
        sys.exit()
        
    # Add log probability of our chosen action to our history
    if type(agent.policy_history) != torch.Tensor:
        agent.policy_history = torch.cat([agent.policy_history, c.log_prob(action)[0][0]])
    else:
        
        agent.policy_history = c.log_prob(action)[0][0]
    return action

def format_action(action_list:list):
    # print(action_in)
    action = {
        "hold": 0,
        "buy": 0,
        "sell": 0
    }
    action_in = action_list.data[0][0]
    
    if action_in == 1:
        action["buy"] = 10
        episode_actions["buy"] += 1
    elif action_in == 2:
        action["sell"] = 10
        episode_actions["sell"] += 1
    else:
        action["hold"] = 1
        episode_actions["hold"] += 1

    return action

def update_agent():
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in agent.reward_episode[::-1]:
        R = r + hparams["gamma"] * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards).cuda()
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    # Calculate loss
    loss = Variable(torch.sum(agent.policy_history * rewards * -1, dim=-1), requires_grad=True) 
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    agent.loss_history.append(loss.item())
    agent.reward_history.append(np.sum(agent.reward_episode))
    agent.policy_history = Variable(torch.Tensor())
    agent.reward_episode= []

agent = Agent().cuda()
if LOAD:
    agent.load_state_dict(torch.load(OUTPUT_PATH))
optimizer = optim.Adam(agent.parameters(), lr=hparams["learning_rate"])

records = open("models/regular/rewards.csv", "w")

if __name__ == "__main__":
    all_balances = []
    
    for episode in range(8000):
        state = env.reset() # Reset environment and record the starting state
        episode_actions = {
            "hold": 0,
            "buy": 0,
            "sell": 0
        }
        
        done = False
        while not done:
            
            # Step through environment using chosen action
            actions = select_action(state.values.reshape(env.observation_space))
            action_dict = format_action(actions)

            state, reward, done = env.step(action_dict)

            agent.reward_episode.append(reward)

        
        update_agent()
        
        

        final_balance = env.portfolio["balance"]
        records.write("{},{:.2f}\n".format(episode, final_balance))

        all_balances.append(final_balance)    

        if episode % 25 == 0:
            print("#" * 60)

            env_change = env.net_change()
            agent_change = (100000 - all_balances[-1]) / 100000
            print("Environment: {} \t Agent: {} \t Difference: {}".format(env_change, agent_change, (agent_change - env_change)/ env_change ))
            print('Episode {}\tLast Balance: {:.2f}\tRunning Reward: {:.2f}'.format(episode, final_balance, np.array(all_balances).mean() ))

            # keys = list(set(agent.reward_episode))
            # freq = {}
            # for key in keys:
            #     freq[key] = 0
            # for val in agent.reward_episode:
            #     freq[val] += 1
            # print(freq)
            print(episode_actions)

            print("#" * 60)
            torch.save(agent.state_dict(), OUTPUT_PATH)

        if np.array(all_balances).mean() > 2000:
            print("Solved! Last Balance is now {} and the average reward is {}".format(final_balance, np.array(all_balances).mean() ))
            break

