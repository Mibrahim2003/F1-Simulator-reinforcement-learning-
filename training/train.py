"""
Training script for F1 Racing RL Agent using Stable-Baselines3.
Implements PPO with parallel environments and TensorBoard logging.
"""
import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import F1RacingEnv


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(rank: int, seed: int = 0, render_mode=None):
    """Create a single environment instance."""
    def _init():
        env = F1RacingEnv(render_mode=render_mode)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def create_callbacks(config: dict, eval_env, log_dir: str, model_dir: str):
    """Create training callbacks."""
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=model_dir,
        name_prefix='f1_racing',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - evaluate on separate env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    return CallbackList(callbacks)


def train(config_path: str, resume_from: str = None):
    """Main training function."""
    print("=" * 60)
    print("F1 Racing RL Training - Stable-Baselines3 PPO")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    print(f"\nLoaded config from: {config_path}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{timestamp}"
    
    log_dir = os.path.join(config['paths']['log_dir'], run_name)
    model_dir = os.path.join(config['paths']['model_dir'], run_name)
    tensorboard_dir = config['paths']['tensorboard_dir']
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    print(f"Run name: {run_name}")
    print(f"Log dir: {log_dir}")
    print(f"Model dir: {model_dir}")
    print(f"TensorBoard: {tensorboard_dir}")
    
    # Create vectorized training environment
    num_envs = config['env']['num_envs']
    print(f"\nCreating {num_envs} parallel environments...")
    
    # Use DummyVecEnv for simplicity (SubprocVecEnv can have issues on Windows)
    train_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    
    # Create evaluation environment (single, non-vectorized)
    eval_env = DummyVecEnv([make_env(0, seed=100)])
    
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
    # Create or load model
    hp = config['hyperparameters']
    policy_config = config['policy']
    
    if resume_from:
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(resume_from, env=train_env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            policy=policy_config['type'],
            env=train_env,
            learning_rate=hp['learning_rate'],
            n_steps=hp['n_steps'],
            batch_size=hp['batch_size'],
            n_epochs=hp['n_epochs'],
            gamma=hp['gamma'],
            gae_lambda=hp['gae_lambda'],
            clip_range=hp['clip_range'],
            ent_coef=hp['ent_coef'],
            vf_coef=hp['vf_coef'],
            max_grad_norm=hp['max_grad_norm'],
            policy_kwargs=dict(net_arch=policy_config['net_arch']),
            tensorboard_log=tensorboard_dir,
            verbose=1
        )
    
    print(f"\nModel architecture:")
    print(f"  Policy: {policy_config['type']}")
    print(f"  Network: {policy_config['net_arch']}")
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_env, log_dir, model_dir)
    
    # Training
    total_timesteps = config['training']['total_timesteps']
    print(f"\n{'='*60}")
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config['training']['log_interval'],
            tb_log_name=run_name,
            reset_num_timesteps=resume_from is None
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, final_model_path


def main():
    parser = argparse.ArgumentParser(description='Train F1 Racing RL Agent')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='training/configs/ppo_default.yaml',
        help='Path to training config YAML'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='Path to model checkpoint to resume from'
    )
    args = parser.parse_args()
    
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
