class Buffer:
    def __init__(self, buffer_capacity, batch_size, num_states, num_actions):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((buffer_capacity, num_states))
        self.action_buffer = np.zeros((buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def sample_batch(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        indices = np.random.choice(record_range, self.batch_size)
        return (
            self.state_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_state_buffer[indices],
        )
