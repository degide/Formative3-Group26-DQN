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

| # | `lr` | `gamma` | `batch_size` | `ε_start` | `ε_end` | `ε_decay (fraction)` | Noted Behaviour |
|---|------|---------|--------------|-----------|---------|----------------------|-----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Baseline config. Most stable performance among all experiments. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Higher LR causes unstable Q-value updates; performance degraded significantly. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | LR too low; learning extremely slow with minimal improvement observed. |
| 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 0.10 | Lower γ makes agent myopic; ignores future rewards, poor survival. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 0.10 | Higher γ enables long-term planning; steadier but slower learning curve. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | Larger batch reduces gradient variance but results in fewer updates overall. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 0.10 | Smaller batch introduces noisy gradients; worst performance observed. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.05 | Fast ε decay leads to premature exploitation with underdeveloped policy. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.20 | Slower ε decay allows more exploration; improved state-space coverage. |
| 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | Combined tuning approach; second-best performance after baseline. |

<<<<<<< HEAD
**Best Configuration:** In Experiment 1 (Baseline), the default hyperparameters provided the most stable learning. Experiment 10 (Combined) showed that moderate tuning can approach baseline performance.
=======
> **Best Configuration:** Experiment 1 (Baseline) — the default hyperparameters provided the most stable learning. Experiment 10 (Combined) showed that moderate tuning can approach baseline performance.

---
>>>>>>> e5b552711529d249f400c2409cce457155bda79d

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
<<<<<<< HEAD
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | MlpPolicy replacing CnnPolicy; fastest run (56s) but worst best eval (259). Raw pixels need convolutions. |

**Best Configuration:** Experiment 2 (High LR) achieved the highest best eval reward (377) with a strong mean reward, suggesting that a moderately higher learning rate accelerates convergence without destabilising training. Experiment 8 (Small Buffer) delivered the highest mean episode reward (255) but completed fewer episodes, indicating faster but potentially less generalisable learning.

## Gameplay Demo

[DEMO](./game_play/game_play.mp4)
=======
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | MlpPolicy replacing CnnPolicy; fastest run (56s) but worst best eval (259) — raw pixels need convolutions. |

> **Best Configuration:** Experiment 2 (High LR) — achieved the highest best eval reward (377) with a strong mean reward, suggesting that a moderately higher learning rate accelerates convergence without destabilising training. Experiment 8 (Small Buffer) delivered the highest mean episode reward (255) but completed fewer episodes, indicating faster but potentially less generalisable learning.
>>>>>>> e5b552711529d249f400c2409cce457155bda79d
