"""
Deep Q-Network (DQN) Evaluation & Gameplay Script
Environment: ALE/SpaceInvaders-v5 (Atari)
Framework: Stable Baselines3 + Gymnasium + ALE

Usage:
    python play.py --model dqn_model.zip --episodes 5 --render
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Disable GPU for evaluation.

import argparse
import time
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


# Configuration
ENV_ID          = "ALE/SpaceInvaders-v5"
N_STACK         = 4       # Same value from train.py
DEFAULT_MODEL   = "best_model/best_model.zip"   # Path to the saved model
N_EPISODES      = 5       # Number of evaluation episodes
RENDER_DELAY    = 0.03    # Seconds between rendered frames (controls visual speed)
VIDEO_FOLDER    = "game_play"                   # Output folder for episode videos


def evaluate_agent(
    model_path: str,
    n_episodes: int,
    render: bool,
) -> None:
    """
    Load a trained DQN model and evaluate it over a specified
    number of episodes using a Greedy Q-Policy (deterministic=True).
    Each episode is recorded and saved as an .mp4 file in the
    VIDEO_FOLDER directory.

    Parameters
    ----------
    model_path : str
        Path to the saved .zip model file.
    n_episodes : int
        Number of complete episodes to run.
    render : bool
        Whether to visualise the game using env.render().
    """

    # Load the trained model
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}\n")

    model = DQN.load(model_path)

    # Create the output folder for videos if it doesn't exist
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    print(f"Videos will be saved to: {os.path.abspath(VIDEO_FOLDER)}\n")

    # Evaluation loop. Record each episode individually so every
    # episode gets its own clearly named video file.
    episode_rewards = []
    episode_lengths = []

    print(f"Running {n_episodes} evaluation episode(s) with Greedy Q-Policy\n")

    for ep in range(1, n_episodes + 1):

        # render_mode must be "rgb_array" for VecVideoRecorder to capture frames.
        # When --render is passed we also open a human window via a separate
        # render() call inside the loop.
        env = make_atari_env(ENV_ID, n_envs=1, seed=ep,
                             env_kwargs={"render_mode": "rgb_array"})
        env = VecFrameStack(env, n_stack=N_STACK)

        # Wrap with the video recorder.
        # video_length is set very large so the recorder never cuts the episode
        # short; we stop it manually after the episode ends.
        video_name_prefix = f"episode_{ep:02d}"
        env = VecVideoRecorder(
            env,
            video_folder=VIDEO_FOLDER,
            record_video_trigger=lambda step: step == 0,  # start at step 0
            video_length=1_000_000,
            name_prefix=video_name_prefix,
        )

        obs          = env.reset()
        done         = False
        total_reward = 0.0
        step_count   = 0

        while not done:
            # Greedy Q-Policy: deterministic=True
            action, _state = model.predict(obs, deterministic=True)

            obs, reward, terminated, info = env.step(action)

            total_reward += float(reward.sum())
            step_count   += 1
            done          = bool(terminated.any())

            if render:
                env.render(mode="human")
                time.sleep(RENDER_DELAY)

        # Closing the VecVideoRecorder flushes and finalises the .mp4 file.
        env.close()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        print(
            f"  Episode {ep:>2}/{n_episodes}  |  "
            f"Reward: {total_reward:8.2f}  |  "
            f"Steps: {step_count:>5}  |  "
            f"Video: {video_name_prefix}-step-0-to-step-{step_count}.mp4"
        )

    # Summary
    print(f"\n{'-'*60}")
    print("EVALUATION SUMMARY")
    print(f"{'-'*60}")
    print(f"  Environment       : {ENV_ID}")
    print(f"  Model             : {model_path}")
    print(f"  Policy mode       : Greedy Q-Policy (deterministic=True)")
    print(f"  Episodes          : {n_episodes}")
    print(f"  Mean Reward       : {np.mean(episode_rewards):.2f}")
    print(f"  Std  Reward       : {np.std(episode_rewards):.2f}")
    print(f"  Max  Reward       : {np.max(episode_rewards):.2f}")
    print(f"  Min  Reward       : {np.min(episode_rewards):.2f}")
    print(f"  Mean Episode Len  : {np.mean(episode_lengths):.1f} steps")
    print(f"  Videos saved to   : {os.path.abspath(VIDEO_FOLDER)}")
    print(f"{'-'*60}\n")


# Command-line argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on an Atari environment."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to trained model .zip (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EPISODES})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the game GUI during evaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
    )