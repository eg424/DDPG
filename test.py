import gymnasium as gym
import tensorflow as tf
from utils.actor_critic import get_actor
import numpy as np

# Load pre-trained models
actor_model = tf.keras.models.load_model('actor_model.h5')

# Set up environment
env = gym.make("Pendulum-v1", render_mode="human")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

def policy(state, noise_object):
    sampled_actions = np.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

# Run the trained agent
total_reward = 0
episodes = 10

ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=0.2 * np.ones(1))

for ep in range(episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)
        action = policy(tf_prev_state, ou_noise)
        state, reward, done, truncated, _ = env.step(action)

        episodic_reward += reward

        if done or truncated:
            break

        prev_state = state

    total_reward += episodic_reward
    print(f"Episode {ep}, Reward: {episodic_reward}")

print(f"Average Reward over {episodes} episodes: {total_reward / episodes}")
