"""
Test script for the F1 Racing Gymnasium Environment.
Runs a random agent and validates the environment with check_env.
"""
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.env import F1RacingEnv


def test_random_agent(num_episodes: int = 3, render: bool = True):
    """Test the environment with a random agent."""
    print("=" * 50)
    print("Testing F1 Racing Environment with Random Agent")
    print("=" * 50)
    
    # Create environment
    render_mode = "human" if render else None
    env = F1RacingEnv(render_mode=render_mode)
    
    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs[:5]}... (first 5 values)")
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Render
            if render:
                env.render()
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, total={total_reward:.1f}, "
                      f"velocity={info['velocity']:.1f}")
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Crashed: {info.get('crashed', False)}")
        print(f"  Laps: {info.get('laps', 0)}")
    
    env.close()
    print("\n" + "=" * 50)


def test_env_compliance():
    """Test environment compliance with Gymnasium API."""
    print("=" * 50)
    print("Running Gymnasium Environment Compliance Check")
    print("=" * 50)
    
    env = F1RacingEnv(render_mode=None)
    
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("\n[OK] Environment passed all compliance checks!")
    except Exception as e:
        print(f"\n[FAIL] Environment check failed: {e}")
    finally:
        env.close()
    
    print("=" * 50)


def test_observation_action_spaces():
    """Test observation and action space properties."""
    print("=" * 50)
    print("Testing Observation and Action Spaces")
    print("=" * 50)
    
    env = F1RacingEnv(render_mode=None)
    
    # Test observation space
    obs, _ = env.reset()
    print(f"\nObservation Space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low[0]}")
    print(f"  High: {env.observation_space.high[0]}")
    print(f"  Actual observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    
    # Test action space
    print(f"\nAction Space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    
    # Test a few steps
    print("\nTesting 10 steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"Observation not in space at step {i}"
        if terminated or truncated:
            obs, _ = env.reset()
    
    print("[OK] All steps produced valid observations!")
    
    env.close()
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    # Default to visual test
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            test_env_compliance()
        elif sys.argv[1] == "--spaces":
            test_observation_action_spaces()
        elif sys.argv[1] == "--no-render":
            test_random_agent(num_episodes=1, render=False)
    else:
        # Run all tests
        test_observation_action_spaces()
        test_env_compliance()
        print("\nNow running visual test with random agent...")
        print("Press Ctrl+C to stop.\n")
        test_random_agent(num_episodes=2, render=True)
