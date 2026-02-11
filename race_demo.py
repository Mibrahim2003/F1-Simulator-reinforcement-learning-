"""
Race Demo - Multi-Agent F1 Racing Demo

Run a multi-agent race with trained AI or simple AI opponents.
Features: countdown, leaderboard, lap timing, visual effects.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pygame
from src.multi_agent_env import MultiAgentRacingEnv


def run_demo(
    num_racers: int = 4,
    total_laps: int = 3,
    use_trained_model: bool = False,
    model_path: str = None,
    ai_only: bool = False,
    track_name: str = None
):
    """Run a multi-agent race demo."""
    print("=" * 60)
    print("F1 RL Racing - Multi-Agent Race Demo")
    print("=" * 60)
    print(f"Racers: {num_racers}")
    print(f"Laps: {total_laps}")
    print(f"Track: {track_name or 'Oval (default)'}")
    print(f"Mode: {'AI Only' if ai_only else 'Player + AI'}")
    print()
    
    # Load trained model if specified
    model = None
    if use_trained_model and model_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Loaded trained model: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Using simple AI for all agents.")
    
    # Create environment
    env = MultiAgentRacingEnv(
        render_mode='human',
        num_agents=num_racers,
        total_laps=total_laps,
        track_name=track_name
    )
    
    if not ai_only:
        print("\nControls:")
        print("  Arrow Keys / WASD: Control primary car (RED)")
        print("  R: Restart race")
        print("  ESC: Quit")
    else:
        print("\nAI Only Mode - Watch the race!")
        print("  R: Restart race")
        print("  ESC: Quit")
    print()
    print("Starting race in 3 seconds...")
    
    obs, info = env.reset()
    running = True
    user_control = not use_trained_model and not ai_only
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("\nRace restarted!")
        
        # Get action for primary agent
        if user_control:
            # Keyboard control
            keys = pygame.key.get_pressed()
            throttle = 0.0
            steering = 0.0
            
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle = 1.0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                throttle = -0.5
            
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steering = -0.8
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steering = 0.8
            
            action = np.array([throttle, steering], dtype=np.float32)
        elif model:
            # Use trained model
            action, _ = model.predict(obs, deterministic=True)
        else:
            # AI mode - use same AI as other agents
            action = env._get_simple_ai_action(0)
        
        # Step environment
        step_result = env.step(action)
        if step_result is None:
            break
            
        obs, reward, terminated, truncated, info = step_result
        
        # Render
        env.render()
        
        # Check race end
        if terminated:
            print("\n" + "=" * 60)
            print("RACE FINISHED!")
            print("=" * 60)
            
            # Print results
            for racer in env.race_manager.get_leaderboard():
                time_str = env.race_manager.format_time(racer.finish_time) if racer.finish_time else "DNF"
                best_str = racer.get_best_lap_str()
                print(f"  P{racer.position}: {racer.name:12} - Time: {time_str} - Best Lap: {best_str}")
            
            print("\nPress R to restart or ESC to quit")
            
            # Wait for restart
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key == pygame.K_r:
                            obs, info = env.reset()
                            waiting = False
                            print("\nNew race started!")
                
                env.render()
                pygame.time.wait(50)
    
    env.close()
    print("\nDemo complete!")


def main():
    parser = argparse.ArgumentParser(description='F1 Racing Multi-Agent Demo')
    parser.add_argument(
        '--racers', '-n',
        type=int,
        default=4,
        help='Number of racers (2-8)'
    )
    parser.add_argument(
        '--laps', '-l',
        type=int,
        default=3,
        help='Number of laps'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to trained model (optional)'
    )
    parser.add_argument(
        '--ai',
        action='store_true',
        help='AI controls all cars (no player control)'
    )
    parser.add_argument(
        '--track', '-t',
        type=str,
        default=None,
        help='Track name from racetrack database (e.g. Monza, Silverstone, Spa)'
    )
    parser.add_argument(
        '--list-tracks',
        action='store_true',
        help='List all available tracks and exit'
    )
    args = parser.parse_args()
    
    # List tracks if requested
    if args.list_tracks:
        from src.track_loader import TrackLoader
        tracks = TrackLoader.list_available_tracks()
        print("Available tracks:")
        for t in tracks:
            print(f"  {t}")
        return
    
    run_demo(
        num_racers=min(8, max(2, args.racers)),
        total_laps=args.laps,
        use_trained_model=args.model is not None,
        model_path=args.model,
        ai_only=args.ai,
        track_name=args.track
    )


if __name__ == "__main__":
    main()
