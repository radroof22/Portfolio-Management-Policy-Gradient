# Imports
import os
import numpy as np
import pandas as pd
from collections import deque


class Environment:
    stock_path = os.getcwd() + "\\Data\\sandp500\\individual_stocks_5yr\\individual_stocks_5yr\\"
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
        """
        Load all the stocks into memory for the class
        - Resets all the user accounts and other information
        - Reads dataframe into `self.df`
        
        Returns:
            - True: all stocks have been used
        """
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
        """ 
        Takes Action and Returns Next State and Reward
            Args:
                - action: int
        """
        self.step_num += 1 # iterate step count
        done = self._next_state() # Iterate to next day
        reward = 0
        # If the agent ran out of money, they are finished
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
        """ 
        Sell Stock and Obtain Reward
        - Liquidates the entire portfolio
        
        Returns:
            - Reward as the difference between the current price
                and the price you are selling at right now
        """
        # Make sure they are not trying to sell nothing
        assert num_to_sell != 0
        
        # If no stocks are owned, don't bother
        if len(self.portfolio) == 0:
            return 0

        reward = 0

        # Latest day price
        curr_price = self._latest_price() 
        
        # Update user Account
        self.agent_balance += float(num_to_sell) * curr_price

        # For each of the entries in profolio
        for entry in self.portfolio:
            # Calculate the reward
            reward += (curr_price - entry[0]) * entry[1]

        return reward
    def _buy(self, num_to_buy):
        """ 
        Purchases Stocks and Places them in Portfolio Class
        - Conducts the sale by updating agent balance
        - Adds the shares to the porfolio

        Args:
            - num_to_buy: The number of shares the bot wants to purchase
        """
        # Make sure they are not trying to buy nothing
        assert num_to_buy != 0

        # Get Latest day prices
        curr_price = self._latest_price() 

        if self.agent_balance - float(num_to_buy) * curr_price < 0:
            return
        
        # Update account balance
        self.agent_balance -= float(num_to_buy) * curr_price

        # Update transaction report for the agent
        self.portfolio.append([curr_price, num_to_buy])

    def _latest_price(self, n=-1):
        """ 
        Get Open Prices of Most Recent Day (Future Day) 
        
        Args:
            n: How many days do you want (-1)
        Returns:
            - latest_price: Open Price on day
        """
        #print(len(self.state))
        return self.state.iloc[n]["open"]

    def _next_state(self):
        """ Gets Next State from Dataset """
        
        # If this is the first step the agent has taken for this episode
        if len(self.df.iloc[self.step_num+1:self.step_num+1+self.HISTORICAL_DAY]) < 30: return True
        if self.step_num >= self.days_for_stock: return True
        self.state = self.df.iloc[self.step_num:self.step_num+self.HISTORICAL_DAY+1]
        return False

    def _reset_state_and_portfolio(self):
        """ Reset Local State Variable """
        self.state = None
        self.portfolio = deque()
    def reset(self):
        """ 
        Reset Agent Balance
        - Resets agents account balancer
        - Sets agents portfolio to empty
        - returns 
        Returns:
            - one last observations
        """
        # Load latest stock data
        self._next_state()

        # Reset agent account
        self.agent_balance = 20000 
        portfolio = None

        state = self.state.diff()
        state.dropna(inplace=True)
        return state

    """ Checkers """
    def _is_portfolio_is_empty(self):
        """ 
        Check if the porfolio is empty
        """
        return len(self.portfolio) == 0
        
