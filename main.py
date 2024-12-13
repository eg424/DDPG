import os
import matplotlib.pyplot as plt
import gymnasium as gym
from utils.ddpg_agent import DDPGAgent
from utils.helper_functions import save_model, plot_rewards
from utils.config import config

if __name__ == "__main__":
    os.makedirs(config["model_dir"], exist_ok=True)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    agent_with_target = DDPGAgent(env, config, use_target_network=True)
    rewards_with_target = agent_with_target.train()
    agent_with_target.save_models(config["model_dir"])

    agent_without_target = DDPGAgent(env, config, use_target_network=False)
    rewards_without_target = agent_without_target.train()

    plot_rewards(rewards_with_target, rewards_without_target)
