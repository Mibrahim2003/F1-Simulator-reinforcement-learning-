"""Quick test of the new reward system."""
from src.env import F1RacingEnv

env = F1RacingEnv(render_mode=None)
obs, info = env.reset()

print(f"Obs shape: {obs.shape}")
print(f"Obs: {obs}")

# Take a few steps
for i in range(5):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"\nStep {i+1}: reward={reward:.3f}")
    print(f"  Components: {info['reward_components']}")

env.close()
print("\nSUCCESS!")
