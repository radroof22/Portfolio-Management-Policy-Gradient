import pandas as pd
import random
import os

class Environment:
    data_dir = "./Data/sandp500/individual_stocks_5yr/Eval/"
    days = 30
    portfolio = {
        "shares": 0,
        "balance": 100000,
    }
    stock_i = 0

    def __init__(self):
        self.stock_list = [s for s in os.walk(self.data_dir)][0][2]
        random.shuffle(self.stock_list)

    def reset(self):
        """
        Usually called before the program runs. During this time, the ith day will be reset
        and the state will be returned for initial reference

        Returns
        ----------
        state : ndarray[self.days * 5]
            The latest stock prices for the timeframe
        """
        try:
            # print(self.stock_list[self.stock_i])
            self.df = pd.read_csv(self.data_dir + self.stock_list[self.stock_i]).drop(["Date", "Name"], axis=1).dropna()
        except IndexError:
            print("Reset Stock List")
            self.stock_i=0
            self.df = pd.read_csv(self.data_dir + self.stock_list[self.stock_i]).drop(["Date", "Name"], axis=1).dropna()
        self.i = 0
        state, _ = self._get_state()
        self.portfolio = {
            "shares": 0,
            "balance": 100000,
        }

        self.stock_i += 1

        return state

    def step(self, action:dict):
        """
        Environment will simulate whatever action the agent takes. 

        Parameters
        ----------
        action : dict(3)
            >>> {"hold":0, "buy": 1, "sell":0}

        Returns
        -------
        state : ndarray[self.days * 5]
            Game state after the action was made
        reward : int
            Reward from whatever action the agent took
        done : bool
            Whether the dataset for steps has been exhausted
        """

        # Create and find reward for action
        reward = None
        if action["buy"] != 0:
            # Buy some shares
            reward = self._buy(action["buy"])
            
        elif action["sell"] != 0:
            # Buy some shares
            reward = self._sell(action["sell"])
        else:
            reward = self.calc_reward()
        
        state, done = self._get_state()
        # print("BALANCE: %s" % self.portfolio["balance"])
        return state, reward, done

    def _get_state(self, move_day:bool=True):
        """
        Returns the latest state in refernce to `self.i`

        Parameters
        ----------
        move_day : bool
            The option to allow the function to iterate 
            to the next day

        Returns
        ---------
        state : ndarray[self.days * 5]
            Game state after the action has been made
        done : bool[true]
            Whether the dataset has been used up or not
        """
        state = self.df.iloc[self.i:self.i+self.days]
        if move_day: self.i += 1
        return state, self.i+self.days +1 >= len(self.df)

    def _buy(self, n_shares:int):
        """
        Checks if the stock can be purchased based on `self.portfolio.balance`
        and then conducts trade based on last closing price

        Parameters
        ----------
        n_shares : int
            Number of stocks to purchase

        Returns
        --------
        reward : float
            The reward is the cost of the trade that the agent made
            This will be negative because of how you pay money to buy stock
        """
        
        curr_price = self._get_state(move_day=False)[0].iloc[-1]["Close"]
        # Check that the total amount of money needed to buy is less 
        # than the amount of money available to the person
        if curr_price * n_shares > self.portfolio["balance"]: return 0

        # Deduct and Update portfolio
        self.portfolio["shares"] += n_shares
        self.portfolio["balance"] -= curr_price * n_shares
        
        return self.calc_reward() 

    def _sell(self, n_shares:int):
        """
        Sell as many shares as the agent wants to by editing 
        `self.portfolio`. Also updates the balance of money for the 
        agent, too.

        Parameters
        ----------
        n_shares : int
            Number of stocks to purchase

        Returns
        --------
        reward : float
            The reward is the profit that just came from the trade
        """
        # Check that the number of shares agent wants to sell
        # is actually owned by the agent
        if n_shares > self.portfolio["shares"]: return 0
        
        curr_price = self._get_state(move_day=False)[0].iloc[-1]["Close"]

        # Deduct and Update portfolio
        self.portfolio["shares"] -= n_shares
        self.portfolio["balance"] += curr_price * n_shares

        return  self.calc_reward() 

    def calc_reward(self):
        """
        Reward is calculated of of the PE ratio
        
        Returns:
        - Reward: int
        """
        return self.portfolio["balance"] + self.portfolio["shares"] * self._get_state(move_day=False)[0].iloc[-1]["Close"]
    
    def net_change(self):
        return (self.df.iloc[-1]["Close"] - self.df.iloc[0]["Close"]) / self.df.iloc[-1]["Close"]

    def get_df(self):
        """
        Returns Dataframe of the Environment Currently

        Returns
        - df: pd.DataFrame
        """
        return self.df

    @property
    def action_space(self):
        return 3 
    @property
    def observation_space(self):
        return (self.days, 5)


if __name__ == "__main__":
    env = Environment()
    state = env.reset()
        