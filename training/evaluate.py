"""
Evaluation script to visualize trained RL agents.
Loads a trained model and runs it in the environment with rendering.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import F1RacingEnv


def evaluate(
    model_path: str,
    num_episodes: int = 5,
    deterministic: bool = True,
    render: bool = True,
    verbose: bool = True
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to run
        deterministic: Use deterministic actions (no exploration)
        render: Render the environment
        verbose: Print detailed statistics
    """
    print("=" * 60)
    print("F1 Racing RL - Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = F1RacingEnv(render_mode=render_mode)
    
    print(f"Render mode: {render_mode}")
    print(f"Deterministic: {deterministic}")
    print(f"Episodes: {num_episodes}")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    episode_laps = []
    crashes = 0
    
    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Track statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_laps.append(info.get('laps', 0))
        if info.get('crashed', False):
            crashes += 1
        
        if verbose:
            status = "CRASHED" if info.get('crashed', False) else "COMPLETED"
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={total_reward:.1f}, Steps={steps}, "
                  f"Laps={info.get('laps', 0)}, Status={status}")
    
    env.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.1f} (+/- {np.std(episode_rewards):.1f})")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Total Laps: {sum(episode_laps)}")
    print(f"Crash Rate: {crashes}/{num_episodes} ({100*crashes/num_episodes:.1f}%)")
    print(f"Best Episode: Reward={max(episode_rewards):.1f}")
    print(f"{'='*60}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'total_laps': sum(episode_laps),
        'crash_rate': crashes / num_episodes
    }


def find_latest_model(models_dir: str = "models") -> str:
    """Find the most recently saved model."""
    if not os.path.exists(models_dir):
        return None
    
    # Look for final models or best models
    candidates = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith('.zip'):
                path = os.path.join(root, f)
                candidates.append((os.path.getmtime(path), path))
    
    if not candidates:
        return None
    
    # Return most recent
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained F1 Racing agent')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to trained model (.zip). If not specified, finds latest.'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=5,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable rendering'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic (non-deterministic) actions'
    )
    args = parser.parse_args()
    
    # Find model
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("ERROR: No model found. Train a model first with:")
            print("  python training/train.py")
            return
        print(f"Using latest model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return
    
    evaluate(
        model_path=model_path,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
