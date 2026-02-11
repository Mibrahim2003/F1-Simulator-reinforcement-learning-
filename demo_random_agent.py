"""Visual demo of the new reward system with random actions."""
from src.env import F1RacingEnv

print("Starting visual demo of V2 reward system...")
print("Watch the HUD to see reward breakdown!")
print("Note: Random agent - will crash quickly but shows direction enforcement")
print()

env = F1RacingEnv(render_mode='human')

for episode in range(3):
    obs, info = env.reset()
    print(f"Episode {episode + 1}")
    
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        env.render()
        done = term or trunc
        steps += 1
    
    print(f"  Steps: {steps}, Reward: {total_reward:.1f}")
    print(f"  Progress: {info['progress']*100:.1f}%, Laps: {info['laps']}")
    print(f"  Crashed: {info.get('crashed', False)}")

env.close()
print("\nDemo complete!")
