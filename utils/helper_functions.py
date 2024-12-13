import os
import imageio
import tensorflow as tf
import numpy as np

def save_model(actor_model, critic_model, directory):
    """
    Save actor and critic models to the specified directory.
    """
    os.makedirs(directory, exist_ok=True)
    actor_model.save(os.path.join(directory, "actor_model.h5"))
    critic_model.save(os.path.join(directory, "critic_model.h5"))

def generate_frames(actor_model, env, num_steps, noise, lower_bound, upper_bound):
    """
    Generate frames for GIF visualization.
    """
    frames = []
    state, _ = env.reset()
    for _ in range(num_steps):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = actor_model(tf_state)
        state, _, terminated, truncated, _ = env.step(action.numpy())
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames

def plot_rewards(rewards_with_target, rewards_without_target):
    """
    Plot learning curves with and without the target network.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_with_target, label="With Target Network")
    plt.plot(rewards_without_target, label="Without Target Network")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Learning Curves")
    plt.show()
