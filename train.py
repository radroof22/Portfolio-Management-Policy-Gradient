import numpy as np
import tensorflow as tf
import os
#from Model import Model
import Model
from Environment import Environment
from Model import graph1

#model = Model()
env = Environment(800)

resume = True
n_batches = 8000000
sessions_with_stock = 80
max_days_trading_for_one_stock = 20

action_mapper = {0: 'buy', 1: 'hold', 2:'sell'}
logs_dir = os.getcwd()+"\\tmp\\Model_V1_0\\"
save_model_path = os.getcwd() + "\\Saved_Models\\V1\\"

tf.reset_default_graph()

with tf.Session(graph=graph1) as sess:
    tf.global_variables_initializer().run()
    if resume:
        checkpoint = tf.train.get_checkpoint_state(save_model_path)
        Model.saver.restore(sess, checkpoint.model_checkpoint_path)
    # TBoard Setup Stuff
    # Create a summary to monitor cost tensor
    tf.summary.scalar("Loss", tf.reduce_mean(Model.loss))
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("Reward", tf.reduce_mean(Model.rewards))
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # TBoard writer
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    # Number of Batches
    for iB in range(n_batches):
        
        # Episode for a Batch
        historical_actions, historical_states, historical_rewards = [], [], []

        for iE in range(sessions_with_stock):
            env
            env.load_stock()


            state, reward, done = env.step()
            state = Model.reshape_state(state)

            # One Episode
            for _ in range(max_days_trading_for_one_stock):
                action = Model.get_action_sample()
                action_prob = sess.run(Model.choice, feed_dict={Model.X: state})
                choice = action_mapper[action_prob]
                if choice == 'buy':
                    action["buy"] = 10
                elif choice == "sell":
                    action["sell"] = 10
                state, reward, done = env.step(action)
                state = Model.reshape_state(state)
                historical_rewards.append(reward); historical_actions.append(action_prob); historical_states.append(np.array(state))
                if done: break
                
            #print(len(historical_rewards))    
        # Compute Discounted Reward
        historical_rewards = Model.discount_normalize_rewards(historical_rewards)
        historical_actions  = np.array(historical_actions)
        historical_states = np.array(historical_states)
        historical_rewards = np.array(historical_rewards)
        
        _, loss, cross, summary = sess.run([Model.optimizer, Model.loss, Model.cross_entropy, merged_summary_op], feed_dict={
            Model.rewards: historical_rewards,
            Model.actions: historical_actions,
            Model.X: Model.reshape_state(historical_states)
        })
        # Write logs
        writer.add_summary(summary, iB)

        # Save Model
        Model.saver.save(sess, save_model_path+"{}".format(np.mean(historical_rewards)), global_step=iB)

        print("Batch {} \t Reward: {} \t Loss: {}".format(iB, np.mean(historical_rewards), loss))

