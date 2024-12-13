
# Deep Deterministic Policy Gradient for Healthcare

This repository contains an implementation of the DDPG algorithm applied to the "Pendulum-v1" environment from the Gymnasium library. DDPG is an actor-critic, model-free, off-policy algorithm used for reinforcement learning tasks, particularly advantageous in those involving continuous action spaces.

## Requirements

Ensure that you have the following dependencies installed:

- Python 3.x
- `tensorflow`
- `numpy`
- `matplotlib`
- Any additional libraries

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

```bash
git clone <https://github.com/eg424/DDPG.git>
cd <ddpg_pendulum.py>
```

2. Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Run the code:

```bash
python DDPG_Pendulum.py
```

The script will train a DDPG agent to balance the Pendulum-v1 environment and display a performance comparison between using a target network and not using a target network.

## Code Overview

### Key Components

1. **DDPG Algorithm**: 
   - **Actor Network**: Learns the policy (what action to take given a state).
   - **Critic Network**: Evaluates the action taken by the actor based on the state.
   - **Replay Buffer**: Stores past experiences (state, action, reward, next state) for training.
   - **Ornstein-Uhlenbeck Noise**: Added to actions for exploration.

2. **Hyperparameters**:
   - `gamma`: Discount factor for future rewards.
   - `tau`: Soft update coefficient for target networks.
   - `critic_lr`: Learning rate for the critic network.
   - `actor_lr`: Learning rate for the actor network.
   - `total_episodes`: Number of episodes for training.

3. **Training Process**:
   - The agent interacts with the Pendulum environment, records experiences, and learns by optimizing both the actor and critic networks.
   - The target networks are slowly updated using the soft update method.

4. **Visualizations**:
   - The performance of the agent is plotted, showing its average episodic reward with and without the target network.

## How It Works

### Training Loop

1. **Initialization**: The environment is created, and the actor, critic, and target networks are initialized.
2. **Action Selection**: The agent selects actions using the actor network, adding noise for exploration.
3. **Experience Replay**: The agent stores its experiences in the buffer and learns from them in batches.
4. **Target Network Update**: The target networks (actor' and critic') are updated after each episode using a soft update.
5. **Evaluation**: The agent's performance is tracked, and rewards are averaged over recent episodes for smoother plots.

### Example Output

![download (5)](https://github.com/user-attachments/assets/332b6eb6-97c6-4318-a4a7-0cd377ed2f94)
![download (4)](https://github.com/user-attachments/assets/2620bfb4-a4b2-4c75-bb07-0423872b1caa)


## Authors

- Erik Garcia Oyono (www.linkedin.com/in/erik-garcia-oyono)
- **Deep Deterministic Policy Gradient**: *How can AI be used in healthcare?* (https://medium.com/@eg424/deep-deterministic-policy-gradient-how-can-ai-be-used-in-healthcare-13ed7ca64ce3)
