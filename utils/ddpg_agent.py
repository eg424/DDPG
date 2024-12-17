import numpy as np
import tensorflow as tf
from utils.actor_critic import get_actor, get_critic
from utils.replay_buffer import Buffer
from utils.target_update import update_target
from utils.OUnoise import OUActionNoise


class DDPGAgent:
    def __init__(self, env, config, use_target_network=True):
        self.env = env
        self.config = config
        self.use_target_network = use_target_network
        self.actor_model = get_actor(env.observation_space.shape[0], env.action_space.high[0])
        self.critic_model = get_critic(env.observation_space.shape[0], env.action_space.shape[0])

        if use_target_network:
            self.target_actor = get_actor(env.observation_space.shape[0], env.action_space.high[0])
            self.target_critic = get_critic(env.observation_space.shape[0], env.action_space.shape[0])
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())
        else:
            self.target_actor = None
            self.target_critic = None

        self.critic_optimizer = tf.keras.optimizers.Adam(self.config["critic_lr"])
        self.actor_optimizer = tf.keras.optimizers.Adam(self.config["actor_lr"])
        self.buffer = Buffer(
            self.config["buffer_capacity"],
            self.config["batch_size"],
            env.observation_space.shape[0],
            env.action_space.shape[0]
        )
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=self.config["std_dev"] * np.ones(1))

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True) if self.use_target_network else self.actor_model(next_state_batch, training=True)
            y = reward_batch + self.config["gamma"] * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def train(self):
        ep_reward_list = []
        avg_reward_list = []

        for ep in range(self.config["total_episodes"]):
            prev_state, _ = self.env.reset()
            episodic_reward = 0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = np.squeeze(self.actor_model(tf_prev_state).numpy() + self.noise())
                legal_action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
                state, reward, done, truncated, _ = self.env.step([legal_action])

                self.buffer.record((prev_state, [legal_action], reward, state))
                episodic_reward += reward

                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_batch()
                self.update(
                    tf.convert_to_tensor(state_batch),
                    tf.convert_to_tensor(action_batch),
                    tf.convert_to_tensor(reward_batch),
                    tf.convert_to_tensor(next_state_batch)
                )

                if self.use_target_network:
                    update_target(self.target_actor, self.actor_model, self.config["tau"])
                    update_target(self.target_critic, self.critic_model, self.config["tau"])

                if done or truncated:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)
            print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")

        return avg_reward_list

    def save_models(self, directory):
        self.actor_model.save(f"{directory}/actor_model.h5")
        self.critic_model.save(f"{directory}/critic_model.h5")
