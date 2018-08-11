from Model import policy, save_model, select_action, finish_episode,  generate_action_call
from Environment import Environment

# Hyperparameters
n_epochs = 500 # How Many Stocks
n_batches = 10 # On Single Stock
max_steps_per_episode = 90
reward_threshold = 300 # Reward needed to have model called `success`

env = Environment(n_epochs)

def main():
    running_reward = 10 # Reward Averager

    for epoch in range(n_epochs):
        state = env.load_stock()
        for _ in range(n_batches):
            env.reset_episode_variables()
            for t in range(max_steps_per_episode):
                action = select_action(state.tail(1)) # Choose Action
                action_call = generate_action_call(action)
                state, reward, done = env.step(action_call) # Take Action
                policy.rewards.append(reward) # Add reward for action to rewards list
                if done:    break # If you ran out of money or input data, close out of the loop
                
            running_reward = running_reward * .99 + t * 0.01 # Reward Average
            finish_episode() # Learn, Optimize, Descend the gradient...other buzzwords
        
        if epoch % 25 == 0:
            print("Epoch {} \t Ran Out of Money: {:5d} \t Average Reward: {:.2f}".format(epoch, t!=89, running_reward))
        
        save_model()
        
        if running_reward > reward_threshold:
            # If model has succeeded and passed the threshold to complete assignment
            print("Solved! Running reward is now {} and the last episode obtained ${} in profit".format(running_reward, t))
            break

if __name__ == "__main__":
    main()
        