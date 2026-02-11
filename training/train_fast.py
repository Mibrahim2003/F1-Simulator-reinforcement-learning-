"""
FAST Training script for F1 Racing RL Agent.
Uses SubprocVecEnv (Multiprocessing) + GPU Encryption.
"""
import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path
import multiprocessing

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env import F1RacingEnv

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_env_fn(seed: int = 0):
    """Factory function for creating environments."""
    def _init():
        env = F1RacingEnv(render_mode=None) # No rendering in training
        env.reset(seed=seed)
        return env
    return _init

def train(config_path: str, resume_from: str = None):
    print("=" * 60)
    print("F1 Racing RL - FAST TRAINING MODE (GPU + Multiprocess)")
    print("=" * 60)
    
    # Check Hardware
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Device: {device.upper()} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU Only'})")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    
    config = load_config(config_path)
    
    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_fast_{timestamp}"
    log_dir = os.path.join(config['paths']['log_dir'], run_name)
    model_dir = os.path.join(config['paths']['model_dir'], run_name)
    tensorboard_dir = config['paths']['tensorboard_dir']
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create Multiprocessed Environments
    num_envs = config['env']['num_envs']
    print(f"\nSpawning {num_envs} parallel environments (SubprocVecEnv)...")
    
    # VecMonitor automatically wraps per-env monitors
    # We use make_vec_env convenience with vec_env_cls=SubprocVecEnv
    train_env = make_vec_env(
        F1RacingEnv,
        n_envs=num_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={'start_method': 'spawn'} # Crucial for Windows + PyGame
    )
    
    # Eval env (Single process)
    eval_env = make_vec_env(F1RacingEnv, n_envs=1, seed=100)
    
    # Model Setup
    hp = config['hyperparameters']
    policy = config['policy']
    
    if resume_from:
        print(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=train_env, device=device)
    else:
        model = PPO(
            policy=policy['type'],
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
            policy_kwargs=dict(net_arch=policy['net_arch']),
            tensorboard_log=tensorboard_dir,
            device=device,
            verbose=1
        )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=max(1, config['training']['eval_freq'] // num_envs),
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config['training']['save_freq'] // num_envs),
        save_path=model_dir,
        name_prefix='f1_fast'
    )
    
    # Train
    print("\nStarting Training...")
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=CallbackList([eval_callback, checkpoint_callback]),
            tb_log_name=run_name,
            reset_num_timesteps=resume_from is None
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        train_env.close()
        eval_env.close()
        model.save(os.path.join(model_dir, "final_model"))
        print("Model saved.")

if __name__ == "__main__":
    # Windows Multiprocessing Support
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/configs/ppo_fast.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    
    train(args.config, args.resume)
