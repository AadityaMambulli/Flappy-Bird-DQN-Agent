# Flappy Bird Deep Q-Network (DQN) Reinforcement Learning Agent

An end-to-end Reinforcement Learning implementation of a Deep Q-Network (DQN) trained to play Flappy Bird, built using **Gymnasium**, **PyTorch**, and **Pygame**.

---

## 1. Overview & Architecture

### Environment Specification
The custom environment is implemented in `env/flappy_bird_env.py` conforming to the standard `gymnasium.Env` interface.

* **State Space (Continuous Box(4,))**:
  1. `bird_y`: Vertical coordinate of the bird center.
  2. `bird_velocity`: Current vertical velocity of the bird.
  3. `horizontal_distance_to_pipe`: Horizontal distance from the bird to the next upcoming pipe.
  4. `vertical_distance_to_gap`: Vertical distance from the bird to the center of the pipe gap.
* **Action Space (Discrete(2))**:
  * `0`: Do nothing (let gravity pull bird downward).
  * `1`: Flap (apply instantaneous upward impulse).
* **Reward Structure**:
  * `+0.01`: Living reward for each step survived.
  * `+1.00`: Reward for passing through a pipe obstacle.
  * `-10.0`: Terminal penalty on collision with pipe, ground, or ceiling.

### Deep Q-Network (DQN)
The agent utilizes a Multi-Layer Perceptron (MLP) architecture defined in `agent/dqn.py`:
* **Input Layer**: 4 neurons (normalized state representation)
* **Hidden Layer 1**: 128 units with ReLU activation
* **Hidden Layer 2**: 128 units with ReLU activation
* **Output Layer**: 2 units corresponding to $Q(s, a_0)$ and $Q(s, a_1)$

### Key Algorithmic Features
* **Experience Replay Buffer (`agent/replay_buffer.py`)**: Stores up to 50,000 transitions `(s, a, r, s', done)` and samples mini-batches uniformly at random to break temporal autocorrelation in sequential game frames.
* **Target Network**: A synchronized duplicate network stabilizes TD-target estimation:
  $$Y_t = R_{t+1} + \gamma (1 - d) \max_{a'} Q_{\text{target}}(S_{t+1}, a')$$
* **Epsilon-Greedy Exploration**: Starts at $\epsilon = 1.0$ and decays down to $\epsilon_{\text{min}} = 0.01$ with a factor of $0.995$ per episode.
* **Huber Loss (Smooth L1)** and **Gradient Clipping** ($\text{max\_norm}=1.0$) for robust gradient updates.

---

## 2. Project Directory Structure

```
RL agent/
├── env/
│   ├── __init__.py
│   └── flappy_bird_env.py       # Custom Gymnasium Flappy Bird environment with Pygame rendering
├── agent/
│   ├── __init__.py
│   ├── dqn.py                   # PyTorch DQN neural network architecture
│   └── replay_buffer.py         # Experience replay buffer with PyTorch tensor sampling
├── models/
│   ├── .gitkeep
│   ├── flappy_dqn.pth           # Main trained model checkpoint (standard requirement)
│   ├── flappy_bird_dqn_best.pth  # Best checkpoint (highest score / rolling average)
│   ├── flappy_bird_dqn_final.pth # Checkpoint saved at end of training
│   └── training_curves.png      # Matplotlib metrics plot (reward, score, loss, epsilon)
├── train.py                     # DQN training pipeline with logging and checkpointing
├── play.py                      # Evaluation and visual gameplay script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 3. Installation & Setup

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\activate
   ```

2. **Verify Dependencies**:
   The required libraries are:
   * `torch`
   * `gymnasium`
   * `pygame`
   * `numpy`
   * `matplotlib`

---

## 4. Usage

### Training the Agent
To start training the DQN agent from scratch:
```powershell
.\.venv\Scripts\python.exe train.py --episodes 1000 --batch_size 64 --lr 0.001
```

#### Available CLI Arguments for `train.py`:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--episodes` | `1000` | Total training episodes |
| `--batch_size` | `64` | Mini-batch sample size from replay buffer |
| `--lr` | `0.001` | Learning rate for Adam optimizer |
| `--gamma` | `0.99` | Future reward discount factor |
| `--buffer_size` | `50000` | Maximum capacity of replay buffer |
| `--warmup_steps` | `500` | Random action exploration steps before learning begins |
| `--epsilon_start` | `1.0` | Initial exploration rate |
| `--epsilon_end` | `0.01` | Final minimum exploration rate |
| `--epsilon_decay` | `0.995` | Multiplicative decay per episode |
| `--target_update_freq` | `10` | Frequency (in episodes) to sync target network |
| `--save_dir` | `models` | Directory to save checkpoints and training plots |
| `--save_freq` | `100` | Interval in episodes to save periodic checkpoints |
| `--render` | `False` | Visualize gameplay during training |

---

### Behavior-Cloned Autonomous Policy

Train a separate neural policy from successful gap-aware teacher demonstrations. This uses the existing 4 → 128 → 128 → 2 network with supervised cross-entropy training and saves to `models/flappy_bc.pth`; it does not overwrite the DQN checkpoints.
```powershell
.\.venv\Scripts\python.exe behavior_clone.py --rollouts 256 --epochs 20
.\.venv\Scripts\python.exe evaluate_behavior_clone.py --model models/flappy_bc.pth --episodes 10
```

Evaluation loads only the saved network and selects `argmax` from normalized observations; the controller is used only to generate training labels.

Launch the rendered autonomous demo with:
```powershell
.\.venv\Scripts\python.exe play_behavior_clone.py
```

The window can be closed to exit. Press `R` after a collision to restart.

### Watching the Agent Play

#### 1. Watch Trained DQN Agent (default: `models/flappy_dqn.pth`):
```powershell
.\.venv\Scripts\python.exe play.py --episodes 5 --fps 30
```

#### 2. Watch Random Baseline Agent:
```powershell
.\.venv\Scripts\python.exe play.py --random --episodes 5 --fps 30
```

#### Available CLI Arguments for `play.py`:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model` | `models/flappy_dqn.pth` | Path to PyTorch model weights |
| `--episodes` | `5` | Number of test episodes to play |
| `--fps` | `30` | Render frame rate |
| `--random` | `False` | Run random baseline instead of loading weights |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |

---

## 5. Verification & Testing

All environment, buffer, and network components have passed rigorous verification:
* **Reset & Step API**: Successfully resets state vector and yields next state, reward, terminated, truncated, and info dict.
* **Observation Vector**: Exactly 4 continuous float32 values representing bird height, bird velocity, horizontal pipe distance, and vertical gap distance.
* **Action Space**: Validated discrete actions `0` (fall with gravity) and `1` (instantaneous upward flap).
* **Reward Mechanics**: Correctly emits `+0.01` alive reward, `+1.0` pipe clearance reward, and `-10.0` termination penalty upon hitting ground, ceiling, or pipes.
* **Replay Buffer**: Verified FIFO deque storage, length tracking, and tensor batch sampling.
* **DQN Architecture**: Confirmed output tensor shape `(batch_size, 2)` representing Q-values for both actions.
