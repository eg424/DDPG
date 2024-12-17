
# Deep Deterministic Policy Gradient for Healthcare

This repository contains an implementation of the DDPG algorithm applied to the "Pendulum-v1" environment from the Gymnasium library. DDPG is an actor-critic, model-free, off-policy algorithm used for reinforcement learning tasks, particularly advantageous in those involving continuous action spaces.

## Requirements

Ensure that you have the following dependencies installed:

- Python 3.x
- `tensorflow`
- `numpy`
- `matplotlib`
- `gymnasium`
- `imageio`
- Any additional libraries listed in `requirements.txt`

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/eg424/DDPG-Pendulum.git
cd DDPG-Pendulum
```

2. Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Run the code to train the agent:

```bash
python main.py
```

The script will train a DDPG agent to balance the Pendulum-v1 environment and display a performance comparison between using a target network and not using a target network.

4. Test the trained agent:

Use `test.py` to evaluate the trained models:

```bash
python test.py
```

## Code Overview

### Key Components

1. **DDPG Algorithm**: 
   - **Actor Network**: Learns the policy (what action to take given a state).
   - **Critic Network**: Evaluates the action taken by the actor based on the state.
   - **Replay Buffer**: Stores past experiences (state, action, reward, next state) for training.
   - **Ornstein-Uhlenbeck Noise**: Added to actions for exploration.

2. **Files**:
   - main.py: Trains the DDPG agent and saves trained models.
   - test.py: Evaluates the trained agent and generates a performance GIF.
   - utils/ddpg_agent.py: Implements the DDPGAgent class, encapsulating training logic.
   - utils/actor_critic.py: Contains actor and critic network definitions.
   - utils/noise.py: Implements the Ornstein-Uhlenbeck noise process.
   - utils/replay_buffer.py: Manages the replay buffer for experience storage and sampling.
   - utils/target_update.py: Implements the target network soft-update logic.
   - utils/helper_functions.py: Helper functions for saving models, plotting rewards, and      generating GIFs.
   - ddpg_pendulum.py: A self-contained script demonstrating the full implementation.
   
3. **Hyperparameters**:
   - `gamma`: Discount factor for future rewards.
   - `tau`: Soft update coefficient for target networks.
   - `critic_lr`: Learning rate for the critic network.
   - `actor_lr`: Learning rate for the actor network.
   - `total_episodes`: Number of episodes for training.

## How It Works

### Training Loop

1. **Initialization**: The environment is created, and the actor, critic, and target networks are initialized.
2. **Action Selection**: The agent selects actions using the actor network, adding noise for exploration.
3. **Experience Replay**: The agent stores its experiences in the buffer and learns from them in batches.
4. **Target Network Update**: The target networks (actor' and critic') are updated after each episode using a soft update.
5. **Evaluation**: The agent's performance is tracked, and rewards are averaged over recent episodes for smoother plots.

### Testing
1. **Load Models**: The trained actor model is loaded from the saved_models/ directory.
2. **Generate GIF**: The trained agent is evaluated in the environment. A sequence of frames is recorded and saved as a GIF in the gifs/ directory.
   
### Example Output
**Learning Curves**:
The agent's average episodic reward over training episodes, comparing performance with and without target networks.
![download (6)](https://github.com/user-attachments/assets/b500efde-8843-49c7-9c45-d52ff39ace6d)

**GIF of Agent Performance**:
A GIF showcasing the trained agent balancing the pendulum.
![pendulum_solved](https://github.com/user-attachments/assets/e089a839-39ca-4c4a-8687-609d4cebd3d7)


## Authors

- Erik Garcia Oyono (www.linkedin.com/in/erik-garcia-oyono)
- **Deep Deterministic Policy Gradient**: *How can AI be used in healthcare?* (https://medium.com/@eg424/deep-deterministic-policy-gradient-how-can-ai-be-used-in-healthcare-13ed7ca64ce3)
