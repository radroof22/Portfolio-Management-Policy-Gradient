import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

state_size = 5
action_size = 3
dropout_amount =.5

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # Archeticture
        self.lstm = nn.LSTM(state_size, 128)
        # Hidden [Length of Sequence, mini batch, inputs of each item]
        self.hidden = (torch.randn(1, 1, 128),
            torch.randn(1, 1, 128))
        self.layer1 = nn.Linear(128, 64)#nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(p=dropout_amount)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout_amount)
        self.layer3 = nn.Linear(32, 64)
        self.dropout3 = nn.Dropout(p=dropout_amount)
        self.layer4 = nn.Linear(64, 16)
        self.layer5 = nn.Linear(16, action_size)

        # Log for Training
        self.episode_logs = {"action":[], "reward": []}
        self.action_mapper = {0:'hold', 1:'buy', 2:'sell'}

        # Optimizer Definition
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.normalize(x)
        # print(x.shape)
        '''for i in x:
            # print(i.shape)
            # Step through sequence @1 time
            # After each step, hidden contains the hiddne state
            # print(i.view(1, 1, state_size).shape)
            # print(self.hidden.shape)
            out, hidden = self.lstm(i.view(-1, 1, state_size), self.hidden)'''
        x = x.view(30, 1, 5)
        out, hidden = self.lstm(x, self.hidden)
        # print(hidden)
        x = F.relu(self.layer1(out[-1]))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.relu(self.layer4(x))
        action_score = F.relu(self.layer5(x))
        # print(action_score  )
        return F.softmax(action_score, dim=-1)

    def select_action(self, x):        
        #print(x)
        y_ = self.forward(x)
        # print(y_)
        m = Categorical(y_) # Create normalized distribution
        action = m.sample() # Choose action
        #_, action_i = y_.max(-1)
        # print(action_i)
        # Log action
        
        self.episode_logs["action"].append(y_)
        return action

    def log_reward(self, reward):
        self.episode_logs['reward'].append(reward)

    def generate_action_call(self,action_key, amount=10):
        # print(action_key.numpy())
        action_val = self.action_mapper[int(action_key.numpy())]
        action = {
            "buy": 0,
            "sell": 0
        }
        if action_val != "hold":
            action[action_val] = amount
        return action

    def optimize(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.episode_logs["reward"][::-1]:
            R = r+.95 * R
            rewards.insert(0,R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps.item())
        for log_prob, reward in zip(self.episode_logs["action"], self.episode_logs["reward"]):
            policy_loss.append(-log_prob * reward)
        
        self.optimizer.step()
        self.episode_logs = {"action":[], "reward": []}
    
    


def train():
    from Environment import Environment
    n_stocks = 25
    policy = Policy()
    env = Environment(n_stocks)
    
    for _ in range(n_stocks): # Stocks to iterate over
        env.load_stock() #Load the stock into the environment
        print("*"*8 + str(env.stock_file) + "*"*8)
        for _ in range(10): # number of session with one stock
            state = env.reset_episode_variables()

            episode_reward = 0
            for _ in range(90): # Days to trade with stock
                action = policy.select_action(state.values)
                action = policy.generate_action_call(action)
                state, reward, done = env.step(action=action)

                policy.log_reward(reward)
                episode_reward += reward

                if done: break
            #print("*"*8, "OPTIMIZING", "*"*8)
            policy.optimize()
            # for i in policy.parameters():
            #     print(i.data)
            print("Episode Reward: {}".format(episode_reward))
        

if __name__ == "__main__":
    train()