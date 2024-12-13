import tensorflow as tf
from tensorflow.keras import layers

def get_actor(num_states, upper_bound):
    """
    Creates the actor model.
    
    :param num_states: Number of state inputs.
    :param upper_bound: Upper bound of the action space.
    :return: Actor model.
    """
    last_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(num_states, num_actions):
    """
    Creates the critic model.
    
    :param num_states: Number of state inputs.
    :param num_actions: Number of action inputs.
    :return: Critic model.
    """
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Combine state and action inputs
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model
