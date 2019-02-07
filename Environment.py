# Imports
import os
import numpy as np
import pandas as pd
from collections import deque


class Environment:
    stock_path = os.getcwd() + "\\Data\\Individual_Stocks_5yr\\"
    agent_balance = 20000
    step_num = 0
    HISTORICAL_DAY = 30
    portfolio = None # of type deqeue
    state = None # Of type Deque
    hold_reward = .01
    days_for_stock = 5000

    def __init__(self, stock_iteration_amounts:int):
        # Construct List of Stocks to iterate over and their files
        # Find all files in the stocks directory (individual stocks)
        stock_list = os.listdir(self.stock_path)
        # Shuffle stocks list
        np.random.shuffle(stock_list)
        # Construct generator for memory management during training process
        self.stock_file_list =  (n for n in stock_list[:stock_iteration_amounts])

    def load_stock(self):
        try:
            # Get Next Entry in File List Generator
            self.stock_file = next(self.stock_file_list)
            # Open Stocks, historical data, and drop [date, name]
            self.df = pd.read_csv(self.stock_path + self.stock_file).drop(["Name", "date"], axis=1)
            
            # Reset episode number because of new episode and dequeu for taking steps in future
            self.step_num = 0
            self.reset()
            self._reset_state_and_portfolio()
        except StopIteration:
            # Finished with all stock datasets that were predefined
            return True
    
    def step(self, action):
        # action={"buy":0, "sell":0}
        """ Takes Action and Returns Next State and Reward """
        self.step_num += 1 # iterate step count
        done = self._next_state() # Iterate to next day
        reward = 0
        if self.agent_balance < 0: done= True
        if not done:
            # Handle Buying
            if action["buy"] > 0:
                reward = 0.1
                self._buy(action["buy"])
            # Create Reward: (Sell Price - Purchase Price) * NumberOfShares
            

            # Handle Selling
            elif action["sell"] > 0: 
                reward = self._sell(action["sell"])
        
        
        state = self.state.diff()
        state.dropna(inplace=True)
        # First In First Out
        return state, reward, done # state, reward, done

    def _sell(self, num_to_sell, c_profit=0):
        """ Sell Stock and Obtain Reward. Sells all of portfolio """
        # If no stocks are owned, don't bother
        if len(self.portfolio) == 0:
            return 0

        reward = 0
        curr_price = self._latest_price() # Latest day price
        
        for entry in self.portfolio:
            reward += (curr_price - entry[0]) * entry[1]

        return reward
    def _buy(self, num_to_buy):
        """ Purchases Stocks and Places them in Portfolio Class """
        stock_price = self._latest_price() # Latest day
        if self.agent_balance - float(num_to_buy) * stock_price < 0:
            return
        # Update account balance
        self.agent_balance -= float(num_to_buy) * stock_price
        # Update transaction report for the agent
        assert num_to_buy != 0
        self.portfolio.append([stock_price, num_to_buy])

    def _latest_price(self, n=-1):
        """ Get Open Prices of Most Recent Day (Future Day)"""
        #print(len(self.state))
        return self.state.iloc[n]["open"]

    def _next_state(self):
        """ Gets Next State from Dataset """
        
        # If this is the first step the agent has taken for this episode
        if len(self.df.iloc[self.step_num+1:self.step_num+1+self.HISTORICAL_DAY]) < 30: return True
        if self.step_num >= self.days_for_stock: return True
        self.state = self.df.iloc[self.step_num:self.step_num+self.HISTORICAL_DAY+1]
        ## if len(self.state) < 30: print("IOGSDYGIOHIOSDGHIODGHIOSDGHIO") 
        return False

    def _reset_state_and_portfolio(self):
        """ Reset Local State Variable """
        self.state = None
        self.portfolio = deque()
    def reset(self):
        """ Reset Agent Balance """
        # Load latest stock data
        self._next_state()
        self.agent_balance = 20000 # Reset agent account
        portfolio = None

        state = self.state.diff()
        state.dropna(inplace=True)
        return state

    """ Checkers """
    def _is_portfolio_is_empty(self):
        return len(self.portfolio) == 0
        
