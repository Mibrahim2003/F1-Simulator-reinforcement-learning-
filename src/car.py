"""
Car class implementing the Dynamic Bicycle Model for realistic F1 vehicle physics.

Uses the F1DynamicsModel (6-DOF) with Pacejka Tire Physics.
"""
import math
import pygame
from dataclasses import dataclass
from src.utils import normalize_angle, clamp
from src.physics_core import F1DynamicsModel, VehicleConfig, VehicleState

@dataclass
class CarConfig:
    """
    Configuration parameters for the car physics.
    Kept for backward compatibility, but values are now mapped to VehicleConfig.
    """
    # Visual Dimensions (used for rendering)
    length: float = 40.0        # Car length in pixels (~5m)
    width: float = 20.0         # Car width in pixels (~2m)
    wheelbase: float = 29.0     # Distance between axles (~3.6m * 8px/m)
    
    # Physics limits (Legacy, mostly unused now as Physics Core handles this)
    max_steering_angle: float = 0.5  # Max steering angle in radians
    steering_speed: float = 3.0
    max_velocity: float = 1200.0     # Max speed for normalization (pixels/s)      

class Car:
    """
    A car using the F1DynamicsModel (Physics Core).
    Wraps the physics engine and handles rendering/inputs.
    """
    
    def __init__(self, x: float, y: float, angle: float = 0.0, config: CarConfig = None):
        """
        Initialize the car.
        
        Args:
            x: Initial x position (pixels)
            y: Initial y position (pixels)
            angle: Initial heading angle in radians
            config: Car visual configuration
        """
        self.config = config or CarConfig()
        
        # Initialize Physics Engine
        # Scale Note: Physics runs in METERS. Rendering runs in PIXELS.
        # Scale Ratio: 8.0 pixels = 1.0 meter
        self.SCALE = 8.0
        
        phys_config = VehicleConfig(
            wheelbase=self.config.wheelbase / self.SCALE
        )
        self.physics = F1DynamicsModel(config=phys_config)
        
        # Set Initial State
        self.physics.state.x = x / self.SCALE
        self.physics.state.y = y / self.SCALE
        self.physics.state.yaw = angle
        
        # Control inputs
        self.throttle_input = 0.0
        self.steering_input = 0.0
        
        # Visual properties
        self.color = (0, 120, 220)
        
        # Smooth steering state for input lag simulation
        self._current_steering = 0.0
        
    @property
    def x(self) -> float:
        return self.physics.state.x * self.SCALE
        
    @x.setter
    def x(self, value):
        self.physics.state.x = value / self.SCALE

    @property
    def y(self) -> float:
        return self.physics.state.y * self.SCALE
        
    @y.setter
    def y(self, value):
        self.physics.state.y = value / self.SCALE

    @property
    def angle(self) -> float:
        return self.physics.state.yaw
        
    @angle.setter
    def angle(self, value):
        self.physics.state.yaw = value

    @property
    def velocity(self) -> float:
        # Return speed in pixels/s for compatibility
        speed_ms = math.sqrt(self.physics.state.vx**2 + self.physics.state.vy**2)
        return speed_ms * self.SCALE * math.copysign(1.0, self.physics.state.vx)

    @velocity.setter
    def velocity(self, value):
        # Approximate setting velocity (body Longitudinal)
        self.physics.state.vx = value / self.SCALE
        self.physics.state.vy = 0 # Assume straight

    @property
    def steering_angle(self) -> float:
        return self._current_steering
    
    @steering_angle.setter
    def steering_angle(self, value):
        self._current_steering = value

    def update(self, dt: float) -> None:
        """
        Update physics for one timestep.
        """
        # Update steering lag
        target_steering = self.steering_input * self.config.max_steering_angle
        
        if abs(self.steering_input) > 0.01:
            steering_diff = target_steering - self._current_steering
            steering_change = self.config.steering_speed * dt
            if abs(steering_diff) < steering_change:
                self._current_steering = target_steering
            else:
                self._current_steering += math.copysign(steering_change, steering_diff)
        else:
            # Return to center
            if abs(self._current_steering) > 0.001:
                return_change = 5.0 * dt
                if abs(self._current_steering) < return_change:
                    self._current_steering = 0.0
                else:
                    self._current_steering -= math.copysign(return_change, self._current_steering)
        
        # Update Physics Engine
        # Automatic Transmission Logic:
        # If Throttle > 0: Accelerate Forward
        # If Throttle < 0:
        #    - If moving forward (vx > 1): Brake
        #    - If stopped/reversing: Reverse (Negative Throttle)
        
        throttle_cmd = 0.0
        brake_cmd = 0.0
        
        # Forward velocity
        vx = self.physics.state.vx
        
        if self.throttle_input > 0:
            throttle_cmd = self.throttle_input
            brake_cmd = 0.0
        elif self.throttle_input < 0:
            # Want to go back/slow down
            if vx > 1.0:
                # Moving forward -> Brake
                brake_cmd = -self.throttle_input
                throttle_cmd = 0.0
            else:
                # Stopped or Reversing -> Reverse Gear
                throttle_cmd = self.throttle_input # Negative throttle
                brake_cmd = 0.0
        
        self.physics.update(dt, self._current_steering, throttle_cmd, brake_cmd)
        
        # Normalize Angle
        self.physics.state.yaw = normalize_angle(self.physics.state.yaw)
    
    def set_controls(self, throttle: float, steering: float) -> None:
        """
        Set control inputs.
        Args:
            throttle: -1 (brake) to 1 (accelerate)
            steering: -1 (left) to 1 (right)
        """
        self.throttle_input = clamp(throttle, -1.0, 1.0)
        self.steering_input = clamp(steering, -1.0, 1.0)
    
    def reset(self, x: float, y: float, angle: float = 0.0) -> None:
        """Reset car to a new position and state."""
        self.physics.state.x = x / self.SCALE
        self.physics.state.y = y / self.SCALE
        self.physics.state.yaw = angle
        self.physics.state.vx = 0.0
        self.physics.state.vy = 0.0
        self.physics.state.yaw_rate = 0.0
        self._current_steering = 0.0
        self.throttle_input = 0.0
        self.steering_input = 0.0
    
    def get_corners(self) -> list[tuple[float, float]]:
        """Get the four corners of the car in world coordinates (pixels)."""
        half_length = self.config.length / 2
        half_width = self.config.width / 2
        
        corners_local = [
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
            (-half_length, -half_width),
        ]
        
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        cx, cy = self.x, self.y
        
        corners_world = []
        for lx, ly in corners_local:
            wx = cx + lx * cos_a - ly * sin_a
            wy = cy + lx * sin_a + ly * cos_a
            corners_world.append((wx, wy))
        
        return corners_world
    
    def render(self, surface: pygame.Surface, debug: bool = False) -> None:
        """Render the car (same as before)."""
        corners = self.get_corners()
        
        pygame.draw.polygon(surface, self.color, corners)
        pygame.draw.polygon(surface, (255, 255, 255), corners, 2)
        
        # Draw heading indicator
        front_center = (
            (corners[0][0] + corners[1][0]) / 2,
            (corners[0][1] + corners[1][1]) / 2
        )
        indicator_length = 15
        indicator_end = (
            front_center[0] + indicator_length * math.cos(self.angle),
            front_center[1] + indicator_length * math.sin(self.angle)
        )
        pygame.draw.line(surface, (255, 200, 0), front_center, indicator_end, 3)
        
        if debug:
            # Draw Velocity Vectors from Physics State
            cx, cy = self.x, self.y
            vx_px = self.physics.state.vx * self.SCALE
            vy_px = self.physics.state.vy * self.SCALE
            
            # Rotate body velocities to world for drawing
            cos_a = math.cos(self.angle)
            sin_a = math.sin(self.angle)
            
            wx = vx_px * cos_a - vy_px * sin_a
            wy = vx_px * sin_a + vy_px * cos_a
            
            if abs(wx) > 1 or abs(wy) > 1:
                pygame.draw.line(surface, (0, 255, 0), (cx, cy), (cx + wx * 0.2, cy + wy * 0.2), 2)

    @property
    def speed_kmh(self) -> float:
        """Speed in km/h based on real physics (m/s * 3.6)."""
        speed_ms = math.sqrt(self.physics.state.vx**2 + self.physics.state.vy**2)
        return speed_ms * 3.6
    
    @property
    def tire_health(self) -> float:
        """Get current tire health (0.0 - 1.0)."""
        return self.physics.tires.degradation

    def add_wear(self, amount: float) -> None:
        """Degrade tires by amount."""
        self.physics.tires.add_wear(amount)

    def reset_tires(self) -> None:
        """Reset tires to 100%."""
        self.physics.tires.reset_wear()
    
    def get_state(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'velocity': self.velocity,
            'steering_angle': self._current_steering,
            'speed_kmh': self.speed_kmh,
            'vx': self.physics.state.vx,
            'vy': self.physics.state.vy
        }
