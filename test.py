import gymnasium as gym
import tensorflow as tf
from utils.helper_functions import generate_frames
from utils.config import config
import imageio

if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    actor_model = tf.keras.models.load_model(f"{config['model_dir']}/actor_model.h5")

    frames = generate_frames(actor_model, env, config["num_steps"], None, env.action_space.low[0], env.action_space.high[0])
    imageio.mimsave("gifs/pendulum_solved.gif", frames, fps=30)
