"""
Deep Q-Network (DQN) Training Script
Environment: ALE/SpaceInvaders-v5 (Atari)
Framework: Stable Baselines3 + Gymnasium + ALE

Runs 10 experiments by sweeping over hyperparameter configurations.
Results are saved per-experiment and summarised in experiment_summary.csv.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU. Comment out to enable GPU if available.

import csv
import time
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
import numpy as np

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

#  GLOBAL CONFIGURATION

SAVE_DIR       = "./"
ENV_ID         = "ALE/SpaceInvaders-v5"
N_ENVS         = 4    # Parallel environments for data collection
N_STACK        = 4    # Stacked frames fed to the CNN
TOTAL_TIMESTEPS = 30_000   # Increase for better convergence (e.g., 1M–10M)

# 10 EXPERIMENT CONFIGURATIONS
EXPERIMENTS = [
    # ── Experiment 1 ── Baseline (default hyperparameters) 
    {
        "name":                    "exp01_baseline",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 2 ── Higher learning rate 
    {
        "name":                    "exp02_high_lr",
        "policy":                  "CnnPolicy",
        "learning_rate":           5e-4,           # ← 1e-4 → 5e-4
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 3 ── Lower learning rate
    {
        "name":                    "exp03_low_lr",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-5,           # ← 1e-4 → 1e-5
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 4 ── Lower discount factor (γ)
    {
        "name":                    "exp04_low_gamma",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.90,           # ← 0.99 → 0.90
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    # Experiment 5 ── Larger batch size 
    {
        "name":                    "exp05_large_batch",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              128,            # ← 32 → 128
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 6 ── Extended exploration (ε decays over 30 % of run) ──
    {
        "name":                    "exp06_long_explore",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.30,           # ← 0.10 → 0.30
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 7 ── Higher minimum exploration rate (ε_end) 
    {
        "name":                    "exp07_high_eps_end",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.10,           # ← 0.01 → 0.10
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 8 ── Smaller replay buffer 
    {
        "name":                    "exp08_small_buffer",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             10_000,         # ← 100 000 → 10 000
        "learning_starts":         1_000,          # ← adjusted so learning_starts < buffer_size
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
    #  Experiment 9 ── More frequent target-network updates 
    {
        "name":                    "exp09_freq_target_update",
        "policy":                  "CnnPolicy",
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  500,            # ← 1 000 → 500
        "train_freq":              4,
    },
    #  Experiment 10 ── MlpPolicy (flattened obs, architecture ablation) ─
    {
        "name":                    "exp10_mlp_policy",
        "policy":                  "MlpPolicy",   # ← CnnPolicy → MlpPolicy
        "learning_rate":           1e-4,
        "gamma":                   0.99,
        "batch_size":              32,
        "exploration_fraction":    0.10,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps":   0.01,
        "buffer_size":             100_000,
        "learning_starts":         10_000,
        "target_update_interval":  1_000,
        "train_freq":              4,
    },
]


# CUSTOM LOGGING CALLBACK 

class TrainingLogger(BaseCallback):
    """
    Lightweight callback that logs episode reward and length to the
    console and appends them to a per-experiment CSV file.
    """

    def __init__(self, log_path: str = "training_log.csv", verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int]   = []

        with open(self.log_path, "w") as f:
            f.write("timestep,mean_reward,mean_ep_length\n")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)

                if self.verbose:
                    print(
                        f"[Step {self.num_timesteps:>8}]  "
                        f"Episode Reward: {ep_rew:7.2f}  |  "
                        f"Episode Length: {ep_len}"
                    )

                with open(self.log_path, "a") as f:
                    f.write(f"{self.num_timesteps},{ep_rew:.4f},{ep_len}\n")
        return True


#  ENVIRONMENT FACTORY 

def make_env(n_envs: int = N_ENVS, seed: int = 42) -> VecFrameStack:
    """
    Build a vectorised, pre-processed Atari environment.

    Applies standard Atari pre-processing:
      • Grayscale + resize to 84×84
      • Frame-skip = 4  (NoFrameskip variant)
      • Frame-stack = 4  (temporal context for the CNN)
    """
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


#  PER-EXPERIMENT TRAINING ROUTINE 

def run_experiment(cfg: dict, exp_index: int, total: int) -> dict:
    """
    Train a single DQN experiment using the supplied hyperparameter
    configuration and return a summary dict for the results table.

    Parameters
    ----------
    cfg : dict
        Hyperparameter configuration (must include a "name" key).
    exp_index : int
        1-based index used for progress reporting.
    total : int
        Total number of experiments (for progress reporting).

    Returns
    -------
    dict
        Summary row for experiment_summary.csv.
    """
    name = cfg["name"]
    exp_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT {exp_index}/{total}  —  {name}")
    print("=" * 70)
    for k, v in cfg.items():
        if k != "name":
            print(f"  {k:<30} = {v}")
    print("=" * 70 + "\n")

    train_env = make_env(n_envs=N_ENVS, seed=42)
    eval_env  = make_env(n_envs=1,      seed=99)

    logger_cb = TrainingLogger(
        log_path=os.path.join(exp_dir, "training_log.csv"),
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        # best_model_save_path is intentionally omitted: saving a "best model"
        # per experiment is misleading because each experiment is independent.
        # The global best across all experiments is saved in main() instead.
        log_path=os.path.join(exp_dir, "eval_logs"),
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix="dqn_checkpoint",
        verbose=1,
    )

    model = DQN(
        policy=cfg["policy"],
        env=train_env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=cfg["exploration_initial_eps"],
        exploration_final_eps=cfg["exploration_final_eps"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        target_update_interval=cfg["target_update_interval"],
        train_freq=cfg["train_freq"],
        optimize_memory_usage=False,
        device="cpu",                   # Change to "auto" if GPU is available
        verbose=1,
        tensorboard_log=os.path.join(exp_dir, "tensorboard_logs"),
    )

    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[logger_cb, eval_cb, checkpoint_cb],
        log_interval=10,
    )
    elapsed = time.time() - start_time

    model_path = os.path.join(exp_dir, "dqn_model")
    model.save(model_path)
    print(f"\n[OK]  Model saved  -> {model_path}.zip")

    train_env.close()
    eval_env.close()

    #  Compute summary statistics
    rewards       = logger_cb.episode_rewards
    mean_rew      = float(np.mean(rewards))  if rewards else 0.0
    max_rew       = float(np.max(rewards))   if rewards else 0.0
    n_eps         = len(rewards)
    # best_mean_reward is updated by EvalCallback after each evaluation rollout
    best_eval_rew = float(eval_cb.best_mean_reward) if eval_cb.best_mean_reward != -np.inf else 0.0

    print(
        f"\n[SUMMARY]  {name}\n"
        f"  Episodes completed    : {n_eps}\n"
        f"  Mean episode reward   : {mean_rew:.2f}\n"
        f"  Max episode reward    : {max_rew:.2f}\n"
        f"  Best eval mean reward : {best_eval_rew:.2f}\n"
        f"  Wall-clock time       : {elapsed:.1f}s\n"
    )

    return {
        "experiment":             name,
        "policy":                 cfg["policy"],
        "learning_rate":          cfg["learning_rate"],
        "gamma":                  cfg["gamma"],
        "batch_size":             cfg["batch_size"],
        "exploration_fraction":   cfg["exploration_fraction"],
        "exploration_final_eps":  cfg["exploration_final_eps"],
        "buffer_size":            cfg["buffer_size"],
        "learning_starts":        cfg["learning_starts"],
        "target_update_interval": cfg["target_update_interval"],
        "train_freq":             cfg["train_freq"],
        "episodes_completed":     n_eps,
        "mean_episode_reward":    round(mean_rew,      4),
        "max_episode_reward":     round(max_rew,       4),
        "best_eval_reward":       round(best_eval_rew, 4),
        "wall_clock_seconds":     round(elapsed,       1),
        # Carry the trained model so main() can save the single global best.
        # This key is stripped before writing to CSV.
        "_model": model,
    }


# MAIN: RUN ALL 10 EXPERIMENTS

def main() -> None:
    summary_path     = os.path.join(SAVE_DIR, "experiment_summary.csv")
    best_model_dir   = os.path.join(SAVE_DIR, "best_model")
    all_results      = []
    best_model_obj   = None   # held in memory; written to disk only after all runs
    global_best_rew  = -np.inf
    global_best_name = None

    for i, cfg in enumerate(EXPERIMENTS, start=1):
        result = run_experiment(cfg, exp_index=i, total=len(EXPERIMENTS))

        # ── Track the best model in memory (no disk write yet) ────────────
        if result["best_eval_reward"] > global_best_rew:
            global_best_rew  = result["best_eval_reward"]
            global_best_name = result["experiment"]
            best_model_obj   = result["_model"]
            print(
                f"[BEST]  New global best in memory -> {global_best_name}  "
                f"(best_eval_reward={global_best_rew:.2f})"
            )

        # Strip the model object before storing / writing to CSV
        csv_result = {k: v for k, v in result.items() if k != "_model"}
        all_results.append(csv_result)

        # Write / update the summary CSV after every experiment so partial
        # results are preserved if the run is interrupted.
        fieldnames = list(all_results[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"[OK]  Summary updated -> {summary_path}\n")

    # Save the single best model AFTER all experiments are complete
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_obj.save(os.path.join(best_model_dir, "best_model"))
    print(f"[OK]  Best model ({global_best_name}) saved to: {best_model_dir}/best_model.zip")

    # Print final leaderboard
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD  (sorted by best eval reward, descending)")
    print("=" * 70)
    ranked = sorted(all_results, key=lambda r: r["best_eval_reward"], reverse=True)
    for rank, r in enumerate(ranked, start=1):
        marker = "  ◀ BEST" if r["experiment"] == global_best_name else ""
        print(
            f"  #{rank:>2}  {r['experiment']:<35}  "
            f"eval={r['best_eval_reward']:7.2f}  "
            f"mean={r['mean_episode_reward']:7.2f}  "
            f"episodes={r['episodes_completed']}{marker}"
        )
    print("=" * 70)
    print(f"\n[DONE]  All {len(EXPERIMENTS)} experiments complete.")
    print(f"        Best model ({global_best_name}) saved to: {best_model_dir}/best_model.zip")
    print(f"        Full summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
