# Formative 3 Deep Q-Network (DQN)

## Environment
**ALE/SpaceInvaders-v5**: The agent controls a spaceship and must eliminate waves of alien invaders while avoiding their projectiles. The reward signal is clear and dense, making it well-suited for observing DQN convergence behaviour.

## Project Structure

```
├── train.py               # DQN training script
├── play.py                # Evaluation / gameplay script  
├── dqn_model.zip          # Final saved model
├── best_model/
│   └── best_model.zip     # Best model (saved by EvalCallback)
├── game_play              # The game play of the model playing with the agent
├── checkpoints/           # Periodic checkpoints during training
├── training_log.csv       # Per-episode reward & length log
├── tensorboard_logs/      # TensorBoard training curves
└── README.md
```

## Installation

```sh
git clone https://github.com/degide/Formative3-Group26-DQN.git

cd Formative3-Group26-DQN

pip install -r requirements.txt

AutoROM --accept-license
```

## Usage

### Training
```sh
python train.py
```

### Evaluation (with GUI rendering)
```sh
python play.py --model best_model/best_model.zip --episodes 5 --render
```

## Hyperparameter Tuning Experiments

### Member: James Jok Dut Akuei

| # | `lr`    | `gamma` | `batch_size` | `ε_start` | `ε_end` | `ε_decay (fraction)` | Episodes | Mean Reward | Max Reward | Best Eval Reward | Noted Behaviour |
|---|---------|---------|--------------|-----------|---------|----------------------|----------|-------------|------------|------------------|-----------------|
| 1 | 1e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 213      | 182.11      | 685.0      | 407.0            | Baseline config. Stable and strong performance. |
| 2 | 5e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 220      | 192.11      | 565.0      | 270.0            | Higher LR, faster learning but less stable. |
| 3 | 1e-5    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 207      | 182.61      | 735.0      | 134.0            | Very low LR, slow learning, poor eval. |
| 4 | 1e-4    | 0.90    | 32           | 1.0       | 0.01    | 0.10                 | 210      | 200.52      | 805.0      | 290.0            | Lower gamma, more short-sighted agent. |
| 5 | 1e-4    | 0.99    | 128          | 1.0       | 0.01    | 0.10                 | 210      | 199.90      | 665.0      | 433.0            | Large batch, steadier but slower updates. |
| 6 | 1e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.30                 | 209      | 189.76      | 745.0      | 313.0            | Longer exploration, improved coverage. |
| 7 | 1e-4    | 0.99    | 32           | 1.0       | 0.10    | 0.10                 | 211      | 180.69      | 685.0      | 364.0            | Higher epsilon end, more exploration. |
| 8 | 1e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 194      | 236.93      | 800.0      | 249.0            | Small buffer, high mean reward, fewer episodes. |
| 9 | 1e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 215      | 172.77      | 665.0      | 194.0            | Frequent target updates, unstable. |
|10 | 1e-4    | 0.99    | 32           | 1.0       | 0.01    | 0.10                 | 226      | 174.16      | 855.0      | 332.0            | MLP policy, fastest run, good max reward. |

**Best Configuration:**
The best evaluation reward was achieved by the baseline configuration (Experiment 1, best eval reward: 407.0), with strong mean and max rewards. Large batch size (Experiment 5) also performed well (best eval: 433.0). MLP policy (Experiment 10) achieved the highest max reward (855.0) but lower mean and eval rewards. Overall, the baseline (CnnPolicy, lr=1e-4, gamma=0.99, batch=32, eps decay=0.10) is recommended for stability and performance.

### Member: Nshimiye Emmy

| # | `lr` | `gamma` | `batch_size` | `ε_start` | `ε_end` | `ε_decay (fraction)` | Noted Behaviour |
|---|------|---------|--------------|-----------|---------|----------------------|-----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Baseline config. Stable learning with consistent reward around 176; best overall stability. |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Higher LR boosted mean reward to 199 and best eval to 377; faster but slightly noisier learning. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Very low LR slowed convergence; competitive mean reward (195) but lowest best eval (285). |
| 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 0.10 | Lower γ increased mean reward (201) but reduced best eval (270); agent became more short-sighted. |
| 5 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.10 | Large batch stabilised gradients but significantly increased wall-clock time (278s); lower peak reward. |
| 6 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.30 | Longer exploration fraction raised max episode reward to 840; broader state-space coverage observed. |
| 7 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.10 | Higher ε_end kept exploration alive longer; reduced best eval (294) due to less exploitation. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Small buffer (10k) forced rapid experience turnover; highest mean reward (255) but fewer episodes completed. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | More frequent target updates (every 500 steps) destabilised training; lowest mean reward (163). |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | MlpPolicy replacing CnnPolicy; fastest run (56s) but worst best eval (259). Raw pixels need convolutions. |

**Best Configuration:** Experiment 2 (High LR) achieved the highest best eval reward (377) with a strong mean reward, suggesting that a moderately higher learning rate accelerates convergence without destabilising training. Experiment 8 (Small Buffer) delivered the highest mean episode reward (255) but completed fewer episodes, indicating faster but potentially less generalisable learning.

### Member: Harerimana Eginde

| # | `lr` | `gamma` | `batch_size` | `ε_start` | `ε_end` | `ε_decay (fraction)` | Noted Behaviour |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | **Huge Buffer (500k):** Reduced overfitting to recent experiences. Yielded solid mean rewards (186.4) and a high peak evaluation (370.0), at the cost of slightly higher memory and wall-clock time (214.5s). |
| **2** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | **Slow Target (5000 steps):** A more stable target network heavily benefited the CNN's convergence. Achieved the highest mean (198.6) and the highest best evaluation reward (406.0) overall. |
| **3** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | **Fast Train (freq=1):** Training every single step drastically increased computation time (365.5s, the highest) without yielding superior rewards (Eval: 285.0). Shows diminishing returns on excessive gradient updates. |
| **4** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | **Delayed Start (50k steps):** Waiting too long to start learning starved the agent of training steps. Resulted in the lowest mean reward (132.1) and a fast, but unproductive, run time (119.5s). |
| **5** | 0.0001 | 0.999 | 32 | 1.0 | 0.01 | 0.1 | **High Gamma (0.999):** Forced the agent to heavily value long-term rewards. Resulted in excellent overall stability (Mean: 194.9) and a very high peak evaluation (371.0). |
| **6** | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | **Tiny Batch (16):** Halving the batch size likely introduced noise into the gradient updates, resulting in mediocre evaluation scores (264.0) and lower overall performance. |
| **7** | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.02 | **Instant Decay (0.02 fraction):** Dropping epsilon too fast forced the agent to stop exploring too early, prematurely converging and stunting its peak performance (Eval: 285.0). |
| **8** | 0.00025| 0.99 | 32 | 1.0 | 0.01 | 0.1 | **Mid LR (0.00025):** Increasing the learning rate caused instability in the network's weight updates, resulting in a noticeably lower best evaluation reward (250.0) compared to the baseline. |
| **9** | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 0.1 | **Constant Explore (ε_end=0.2):** Keeping the final exploration rate at 20% crippled the exploitation phase. The agent acted randomly too often, yielding the worst evaluation reward (151.0). |
| **10**| 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | **CNN Optimized (Batch 64):** Doubling the batch size provided highly consistent, stable learning (Mean: 192.9), though its absolute peak (270.0) didn't match the slower-target approach. |

**Best Configuration:** Experiment 2 (Slow Target) achieved the highest best_eval_reward at 406.0, meaning the final policy it learned outperformed all other configurations during evaluation. It secured the highest mean_episode_reward at 198.6, showing that it wasn't just a lucky run, but consistently performed well across episodes.

## Gameplay Demo

[DEMO](./game_play/game_play.mp4)
