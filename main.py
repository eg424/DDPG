import gymnasium as gym
from utils.actor_critic import get_actor, get_critic
from utils.noise import OUActionNoise
from utils.replay_buffer import Buffer
from utils.target_update import update_target
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Environment Setup
env = gym.make("Pendulum-v1", render_mode="rgb_array")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# Hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=std_dev * np.ones(1))

actor_model = get_actor(num_states, upper_bound)
critic_model = get_critic(num_states, num_actions)
target_actor = get_actor(num_states, upper_bound)
target_critic = get_critic(num_states, num_actions)

# Initialize target networks with the same weights as original
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Optimizers
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

buffer = Buffer(buffer_capacity=50000, batch_size=64)

# Training Loop
total_episodes = 100
gamma = 0.99
tau = 0.005

ep_reward_list = []

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)
        action = policy(tf_prev_state, ou_noise)
        state, reward, done, truncated, _ = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    print(f"Episode {ep}, Reward: {episodic_reward}")

# Save model (Optional)
actor_model.save('actor_model.h5')
critic_model.save('critic_model.h5')

# Plot results
plt.plot(ep_reward_list)
plt.title('Training Progress')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()
