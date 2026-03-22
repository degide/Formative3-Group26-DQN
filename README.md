# Formative 3 Deep Q-Network (DQN)

## Environment
**ALE/SpaceInvaders-v5**: The agent controls a spaceship and must eliminate waves of alien invaders while avoiding their projectiles. The reward signal is clear and dense, making it well-suited for observing DQN convergence behaviour.

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

> **Best Configuration:** Experiment 1 (Baseline) — the default hyperparameters provided the most stable learning. Experiment 10 (Combined) showed that moderate tuning can approach baseline performance.