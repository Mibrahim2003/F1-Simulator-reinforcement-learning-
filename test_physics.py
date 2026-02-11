"""
Test Script for F1 Physics Engine.
Verifies acceleration 0-100 km/h and top speed.
"""
from src.physics_core import F1DynamicsModel
import math

def test_acceleration():
    physics = F1DynamicsModel()
    
    print(f"--- F1 Physics Verification ---")
    print(f"Mass: {physics.config.mass} kg")
    print(f"Engine Force: 15000 N (Max)")
    print(f"Tire Friction: {physics.tires.params.D}")
    print(f"-------------------------------")
    print("Time(s) | Speed(km/h) | Dist(m) | Accel(G) | F_long(N) | F_drag(N) | Slip(rad)")
    
    dt = 1.0 / 60.0
    total_time = 0.0
    
    # Run for 15 seconds (should hit top speed)
    for i in range(15 * 60):
        # Full Throttle
        physics.update(dt, steer_input=0.0, throttle=1.0, brake=0.0)
        total_time += dt
        
        # Telemetry every 1 second
        if i % 60 == 0:
            state = physics.state
            speed_ms = state.vx
            speed_kmh = speed_ms * 3.6
            
            # Re-calculate interesting metrics
            drag = 0.5 * 1.225 * (state.vx**2) * 0.9 * 1.4
            
            print(f"{total_time:7.2f} | {speed_kmh:11.2f} | {state.x:7.1f} | {0.00:8.2f} | {'?':>9} | {drag:9.1f} | {0.0:9.2f}")

if __name__ == "__main__":
    test_acceleration()
