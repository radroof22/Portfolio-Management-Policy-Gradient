import torch

from Environment import Environment
from regular_policy import Agent, select_action, format_action
import matplotlib.pyplot as plt

env = Environment()

# CONSTANTS
MODEL_PATH = "models/regular/model_proper_portfolio_value.pt"
N_TESTS = 2

agent = Agent().cuda()
agent.load_state_dict(torch.load(MODEL_PATH))

if __name__ == "__main__":
    history = []
    
    ## Graph Setup
    state = env.reset() 
    df = env.get_df()
    x_axis = [i for i in range(len(df))]
    buy_line = {}
    sell_line = {}

    for episode in range(1):
        print("Episode {}".format(episode))
        
        episode_actions = {
            "hold": 0,
            "buy": 0,
            "sell": 0
        }
        
        done = False
        step = 30
        while not done:
            
            # Step through environment using chosen action
            actions = select_action(state.values.reshape(env.observation_space))
            action_dict = format_action(actions, monitor=False)

            state, reward, done = env.step(action_dict)

            agent.reward_episode.append(reward)
            if action_dict["sell"] != 0:
                sell_line[step] = state.iloc[-1]["close"]
            if action_dict["buy"] != 0:
                buy_line[step] = state.iloc[-1]["close"]
            step += 1

        env_change = env.net_change()
        cash_change = (reward - 100000 ) / 100000
        history.append((reward, env_change, cash_change))

    print(buy_line)
    print(sell_line)
    print("Final Balance: {}".format(history))
    print(env.portfolio)
    # plt.plot([i for i in range(len(df))], df["open"] , color="green", label="Open")
    # plt.plot([i for i in range(len(df))], df["high"] , color="blue", label="High")
    # plt.plot([i for i in range(len(df))], df["low"] , color="yellow", label="Low")
    plt.title("Bot Trades")
    plt.plot(list(buy_line.keys()), list(buy_line.values()) , 'go-', label="Buy")
    plt.plot(list(sell_line.keys() ), list(sell_line.values()), 'ro-', label="Sell")
    plt.plot(x_axis, df["close"] , color="black", label="Close")
    plt.show()
        

