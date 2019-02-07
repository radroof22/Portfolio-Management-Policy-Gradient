import numpy as np
import tensorflow as tf
from Environment import Environment

class PolicyGradientAgent:
    def __init__(self, hparams, sess):
        self._s = sess
        self._input = tf.placeholder(tf.float32, 
            shape=[None, hparams["input_size"]])
        
        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams["hidden_size"],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal
        )

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams["num_actions"],
            activation_fn=None
        )

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # get log probabilities
        log_prob = tf.log(tf.nn.softmax(logits))

        # training part of graph
        self._actions = tf.placeholder(tf.int32)
        self._rewards = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._actions
        action_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.mul(action_prob, self._rewards))

        # update
        optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(loss)

    def act(self, obsv):
        # get on action, by sampling
        self._s.run(self._sample, feed_dict={self._input: [obsv]})

    def train_step(self, obsv, actions, rewards):
        batch_feed = {
            self._input: obsv,
            self._actions: actions,
            self._rewards: rewards
        }
        self._s.run(self._train, feed_dict=batch_feed)

def policy_rollout(env, agent):
    """
    Runs one episode of the environment
    """
    obsv, reward, done = env.reset(), 0, False
    observations, actions, rewards = [], [], []

    while not done:
        observations.append(obsv)

        action = agent.act(obsv)
        obsv, reward, done = env.step(action)
        
        actions.append(action)
        rewards.append(reward)

def process_rewards(rews):
    """
    Convert rewards to rewards for one episode
    """
    # total reward: length of episode
    return [len(rews)] * len(rews)

if __name__ == "__main__":
    env = Environment(30)

    # Define hyperparameters
    hparams = {

    }
