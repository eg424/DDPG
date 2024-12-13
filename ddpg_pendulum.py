# Install necessary packages
!pip install gymnasium pybullet tensorflow imageio

import os
import tensorflow as tf
from keras import layers
import keras
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from IPython import get_ipython
from IPython.display import display

# Configuration (using a dictionary for now)
config = {
    "buffer_capacity": 50000,
    "batch_size": 64,
    "std_dev": 0.2,
    "critic_lr": 0.002,
    "actor_lr": 0.001,
    "total_episodes": 100,
    "gamma": 0.99,
    "tau": 0.005,
    "model_dir": "/content/drive/MyDrive/trained_ddpg_agents",
}

# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
env = gym.make("Pendulum-v1", render_mode="rgb_array")

num_steps = 100

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
# TensorFlow to build a static graph out of the logic and computations in our function.
# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        target_actor,
        target_critic,
        critic_model,
        critic_optimizer,
        actor_model,
        actor_optimizer,
        gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))


        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, target_actor, target_critic, critic_model, critic_optimizer, actor_model, actor_optimizer, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch, target_actor, target_critic, critic_model, critic_optimizer, actor_model, actor_optimizer, gamma)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


class DDPGAgent:
    def __init__(self, env, config, use_target_network=True):
        self.env = env
        self.config = config
        self.use_target_network = use_target_network  # Store whether to use target network
        self.actor_model = get_actor()
        self.critic_model = get_critic()

        if self.use_target_network:
            self.target_actor = get_actor()
            self.target_critic = get_critic()

            # Making the weights equal initially
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())
        else:
            self.target_actor = None  # Set target networks to None if not used
            self.target_critic = None

        self.critic_optimizer = keras.optimizers.Adam(self.config["critic_lr"])
        self.actor_optimizer = keras.optimizers.Adam(self.config["actor_lr"])
        self.buffer = Buffer(self.config["buffer_capacity"], self.config["batch_size"])
        self.ou_noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(self.config["std_dev"]) * np.ones(1)
        )

    def train(self):
        ep_reward_list = []
        avg_reward_list = []

        for ep in range(self.config["total_episodes"]):
            prev_state, _ = self.env.reset()
            episodic_reward = 0

            while True:
                tf_prev_state = keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(prev_state), 0
                )

                action = policy(
                    tf_prev_state,
                    self.ou_noise,
                    self.actor_model,
                    lower_bound,
                    upper_bound,
                )
                state, reward, done, truncated, _ = self.env.step(action)

                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                self.buffer.learn(
                    self.target_actor if self.use_target_network else self.actor_model,  # Use actor_model if target_actor is None
                    self.target_critic if self.use_target_network else self.critic_model, # Use critic_model if target_critic is None
                    self.critic_model,
                    self.critic_optimizer,
                    self.actor_model,
                    self.actor_optimizer,
                    self.config["gamma"]
                )

                if self.use_target_network:  # Update target networks only if use_target_network is True
                    update_target(self.target_actor, self.actor_model, self.config["tau"])
                    update_target(
                        self.target_critic, self.critic_model, self.config["tau"]
                    )

                if done or truncated:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

        # Calculate and return the average reward and standard deviation
        avg_rewards = np.mean(ep_reward_list)
        std_rewards = np.std(avg_rewards)
        return avg_reward_list, std_rewards

    def save_model(self, directory):
        """
        Save the agent's models to the specified directory.

        :param directory: The directory where the models will be saved.
        :return: None
        """
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define file paths for actor and critic models
        actor_path = os.path.join(directory, "actor.pth")
        critic_path = os.path.join(directory, "critic.pth")

        # Save the models using PyTorch's save function
        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.critic_model.state_dict(), critic_path)

        print(f"Models saved to:\nActor: {actor_path}\nCritic: {critic_path}")


#Function to generate a sequence of images for the GIF
def generate_frames(model, env, num_steps, ou_noise, lower_bound, upper_bound):  # Add necessary arguments
    frames = []
    state, _ = env.reset()
    for _ in range(num_steps):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy(tf_state, ou_noise, model, lower_bound, upper_bound)  # Pass all required arguments to policy
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())  # Capture the rendered frame
        if terminated or truncated:
            break  # Stop if the episode ends
    return frames

# Create and train agents
agent_with_target = DDPGAgent(env, config, use_target_network=True)
avg_rewards_1, std_rewards_1 = agent_with_target.train()

agent_without_target = DDPGAgent(env, config, use_target_network=False)
avg_rewards_2, std_rewards_2 = agent_without_target.train()

# Generate the frames
frames = generate_frames(agent_with_target.actor_model, env, num_steps, agent_with_target.ou_noise, lower_bound, upper_bound)  # Pass necessary arguments when calling generate_frames

# Calculate variations
std_rewards_1 = np.array([np.std(avg_rewards_1[max(0, i - 40):i + 1]) for i in range(len(avg_rewards_1))])
std_rewards_2 = np.array([np.std(avg_rewards_2[max(0, i - 40):i + 1]) for i in range(len(avg_rewards_2))])

# Plot the learning curves
plt.figure(figsize=(12, 6))
plt.plot(avg_rewards_1, label="With Target Network")
plt.plot(avg_rewards_2, label="Without Target Network")

plt.fill_between(
    range(config["total_episodes"]),
    np.array(avg_rewards_1) - std_rewards_1,
    np.array(avg_rewards_1) + std_rewards_1,
    alpha=0.2,
    )

plt.fill_between(
    range(config["total_episodes"]),
    np.array(avg_rewards_2) - std_rewards_2,
    np.array(avg_rewards_2) + std_rewards_2,
    alpha=0.2,
    )


plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.title("Learning Curves Comparison")
plt.legend()
plt.show()

# Generate and save GIF
frames = generate_frames(
    agent_with_target.actor_model,
    env,
    num_steps,
    agent_with_target.ou_noise,
    lower_bound,
    upper_bound,
    )
imageio.mimsave("pendulum_solved.gif", frames, fps=30)


