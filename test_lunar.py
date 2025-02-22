import gym
import keras
import numpy as np
# from keras.src.legacy.saving import legacy_h5_format

from keras.models import load_model

env = gym.make("LunarLander-v2", render_mode = "human")
state,_ = env.reset()
state_size = env.observation_space.shape[0]

# my_agent = legacy_h5_format.load_model_from_hdf5("train_lunar_lander.h5", custom_objects={'mse': 'mse'})
my_agent = load_model("train_lunar_lander_1100.h5", custom_objects={'mse': 'mse'})

total_reward = 0

while True:
    env.render()
    state = state.reshape((1,state_size))
    q_values = my_agent.predict(state, verbose=0)
    max_q_values = np.argmax(q_values)

    next_state, reward, terminal, _, _ = env.step(action=max_q_values)
    if terminal:
        break
    total_reward += reward
    state = next_state