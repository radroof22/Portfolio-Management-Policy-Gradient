import numpy as np
import tensorflow as tf
import os

#from Model import Model
import Model
from Environment import Environment
from Model import graph1

testing_episodes = 20
action_mapper = {0: 'buy', 1: 'hold', 2:'sell'}
save_model_path = os.getcwd() + "\\Saved_Models\\V1\\"

#model = Model()
env = Environment(testing_episodes)

tf.reset_default_graph()

with tf.Session(graph=graph1) as sess:
    checkpoint = tf.train.get_checkpoint_state(save_model_path)
    Model.saver.restore(sess, checkpoint.model_checkpoint_path)

    for episode in range(testing_episodes):
        env.load_stock()

        # Episode for a Batch
        historical_actions, historical_states, historical_rewards = [], [], []

        state, reward, done = env.step()
        state = Model.reshape_state(state)
        print(env.stock_file)
        # One Episode
        while True:
            action = Model.get_action_sample()
            action_prob = sess.run(Model.choice, feed_dict={Model.X: state})
            choice = action_mapper[action_prob]
            if choice == 'buy':
                action["buy"] = 10
            elif choice == "sell":
                action["sell"] = 10
            state, reward, done = env.step(action)
            print(state.tail(1))
            print(choice)
            state = Model.reshape_state(state)
            historical_rewards.append(reward); historical_actions.append(action_prob); historical_states.append(np.array(state))
            if done: break
        
        historical_actions  = np.array(historical_actions)
        historical_states = np.array(historical_states)
        historical_rewards = np.array(historical_rewards)
        unique, counts = np.unique(historical_actions, return_counts=True)
        print(dict(zip(unique, counts)))
        print("Episode: {} \t Reward: {} \t Actions: {} \t Final_Account_Balance: {}".format(episode, np.mean(historical_rewards),None,env.agent_balance))