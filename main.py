import os
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

if tf.test.gpu_device_name():
    print("Using GPU:", tf.test.gpu_device_name())
else:
    print("No GPU found. Using CPU.")

import gym
import numpy as np
from collections import deque
# from keras.api.models import Sequential
# from keras.api.layers import Dense
# from keras.api.optimizers import Adam
# from keras.api.layers import Input
from keras.models import Sequential
from keras.layers import  Dense
from keras.optimizer_v2.adam import Adam
from keras.layers import Input
class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = deque(maxlen = 100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_target_nn = 10

        self.main_network = self.get_nn()
        self.target_network = self.get_nn()

        self.target_network.set_weights(self.main_network.get_weights())

    def get_nn(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))

        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)

        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = np.array([batch[2] for batch in exp_batch])
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = np.array([batch[4] for batch in exp_batch])

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)
        q_values = self.main_network.predict(state_batch, batch_size=batch_size,verbose = 0)

        next_q_values = self.target_network.predict(next_state_batch, batch_size=batch_size,verbose = 0)
        max_next_q_values = np.amax(next_q_values, axis=1)

        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q_values[i]
            q_values[i][action_batch[i]] = new_q_values

        self.main_network.fit(state_batch, q_values, verbose=0)

    def make_decision(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)

        return np.argmax(q_values[0])

env = gym.make("LunarLander-v2")
state, _ = env.reset()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print((action_size, state_size))
n_espisodes = 500
n_timesteps = 500
batch_size = 64

my_agent = MyAgent(state_size, action_size)
total_time_step = 0
max_reward = -50

for ep in range(n_espisodes):
    ep_rewards = 0
    state, _ = env.reset()

    while True:
        total_time_step += 1
        if total_time_step % my_agent.update_target_nn == 0:
            my_agent.target_network.set_weights(my_agent.main_network.get_weights())
            total_time_step -= my_agent.update_target_nn
        action = my_agent.make_decision(state)
        next_state, reward, terminal, _, _ = env.step(action)
        my_agent.save_experience(state, action, reward, next_state, terminal)

        state = next_state
        ep_rewards += reward

        # if ep <= 100:
        #     if ep_rewards > max_reward:
        #         my_agent.target_network.set_weights(my_agent.main_network.get_weights())
        #         max_reward = ep_rewards
        # else:
        #     if total_time_step % my_agent.update_target_nn == 0 and ep_rewards > 0:
        #         my_agent.target_network.set_weights(my_agent.main_network.get_weights())

        if terminal:
            print("Ep ", ep + 1, " reach terminal with reward: ", ep_rewards)
            if ep_rewards > max_reward:
                my_agent.target_network.set_weights(my_agent.main_network.get_weights())
                max_reward = ep_rewards
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)
    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay

    if (ep + 1) % 100 == 0:
        filename = f"train_lunar_lander_{ep + 1}.h5"  # Using f-string
        my_agent.main_network.save(filename)

# my_agent.main_network.save("train_lunar_lander.h5")