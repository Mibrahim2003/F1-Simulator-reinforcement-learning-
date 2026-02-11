
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.multi_agent_env import MultiAgentRacingEnv

def test_monza_progress():
    print("Testing Monza Track Progress...")
    env = MultiAgentRacingEnv(
        track_name='Monza',
        num_agents=4,
        render_mode=None  # Headless
    )
    
    obs, info = env.reset()
    print("Environment reset complete.")
    
    # Run for 100 steps with simple AI
    print("Running 100 steps...")
    for i in range(100):
        # AI actions are handled internally if we pass empty actions or use the loop differently
        # But MultiAgentRacingEnv.step expects action for agent 0
        # If we want all AI, we can just pass zeros and let the internal AI take over if configured,
        # OR we can manually call _get_simple_ai_action
        
        # Actually MultiAgentRacingEnv step takes action for agent 0.
        # But wait, race_demo.py handles AI by getting actions from `_get_simple_ai_action`
        # Let's see how race_demo does it.
        
        # In race_demo.py:
        # if ai_only:
        #     action = env._get_simple_ai_action(0)
        # ...
        # agent_actions = {i: env._get_simple_ai_action(i) for i in range(1, num_racers)}
        # obs, reward, terminated, truncated, info = env.step(action, agent_actions)
        
        actions = {}
        for agent_id in range(4):
            actions[agent_id] = env._get_simple_ai_action(agent_id)
            
        action_0 = actions[0]
        other_actions = {k: v for k, v in actions.items() if k != 0}
        
        obs, reward, terminated, truncated, info = env.step(action_0, other_actions)
        
        if i % 20 == 0:
            car0 = env.cars[0]
            calc = env.reward_calculators[0]
            tracker = calc.progress_tracker
            wp_idx, dist = tracker.get_nearest_waypoint(car0.x, car0.y)
            expected = tracker.get_expected_heading(car0.x, car0.y)
            sensor = env.sensors[0]
            lidar = sensor.update(car0.x, car0.y, car0.angle, env.track)
            mid = len(lidar) // 2
            
            print(f"Step {i}:")
            print(f"  Pos: ({car0.x:.1f}, {car0.y:.1f}) Angle: {car0.angle:.2f}")
            print(f"  Nearest WP: {wp_idx} Dist: {dist:.1f} ExpHeading: {expected:.2f}")
            print(f"  LIDAR Front: {lidar[mid]:.2f} Left: {min(lidar[:mid]):.2f} Right: {min(lidar[mid+1:]):.2f}")
            print(f"  Speed: {car0.velocity:.1f}")
            
    print("Test Complete.")

if __name__ == "__main__":
    test_monza_progress()
