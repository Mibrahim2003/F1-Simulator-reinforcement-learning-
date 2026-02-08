"""
Car class implementing the Kinematic Bicycle Model for realistic 2D vehicle physics.

The bicycle model treats the car as having two wheels (front and rear) on a single axis,
providing a good balance between realism and computational simplicity.

Physics Equations:
    x' = v * cos(θ)
    y' = v * sin(θ)  
    θ' = (v / L) * tan(δ)
    
Where:
    (x, y) = position
    θ = heading angle (radians)
    v = velocity
    L = wheelbase (distance between front and rear axles)
    δ = steering angle
"""
import math
import pygame
from dataclasses import dataclass
from src.utils import normalize_angle, clamp


@dataclass
class CarConfig:
    """Configuration parameters for the car physics."""
    # Dimensions
    length: float = 40.0        # Car length in pixels
    width: float = 20.0         # Car width in pixels
    wheelbase: float = 30.0     # Distance between axles
    
    # Physics limits
    max_velocity: float = 300.0      # Max speed (pixels/second)
    max_steering_angle: float = 0.6  # Max steering angle (radians, ~34 degrees)
    
    # Acceleration/deceleration
    acceleration: float = 200.0      # Acceleration rate (pixels/second^2)
    braking: float = 300.0           # Braking deceleration
    friction: float = 50.0           # Natural deceleration when no input
    
    # Steering dynamics
    steering_speed: float = 3.0      # How fast steering wheel turns (radians/second)
    steering_return_speed: float = 5.0  # How fast steering returns to center


class Car:
    """
    A car with Kinematic Bicycle Model physics.
    
    The car maintains a state vector [x, y, theta, velocity] and updates
    it based on control inputs (throttle, steering).
    """
    
    def __init__(self, x: float, y: float, angle: float = 0.0, config: CarConfig = None):
        """
        Initialize the car.
        
        Args:
            x: Initial x position
            y: Initial y position  
            angle: Initial heading angle in radians (0 = facing right)
            config: Car configuration parameters
        """
        self.config = config or CarConfig()
        
        # State vector
        self.x = x
        self.y = y
        self.angle = angle  # Heading angle in radians
        self.velocity = 0.0
        
        # Current steering angle (separate from heading)
        self.steering_angle = 0.0
        
        # Control inputs (set by keyboard or RL agent)
        self.throttle_input = 0.0   # -1 (brake) to 1 (accelerate)
        self.steering_input = 0.0   # -1 (left) to 1 (right)
        
        # Visual properties
        self.color = (0, 120, 220)  # Nice blue color
        
    def update(self, dt: float) -> None:
        """
        Update physics for one timestep using the Kinematic Bicycle Model.
        
        Args:
            dt: Delta time in seconds
        """
        # Update steering angle based on input
        target_steering = self.steering_input * self.config.max_steering_angle
        
        if abs(self.steering_input) > 0.01:
            # Steer towards target
            steering_diff = target_steering - self.steering_angle
            steering_change = self.config.steering_speed * dt
            if abs(steering_diff) < steering_change:
                self.steering_angle = target_steering
            else:
                self.steering_angle += math.copysign(steering_change, steering_diff)
        else:
            # Return to center when no input
            if abs(self.steering_angle) > 0.01:
                return_change = self.config.steering_return_speed * dt
                if abs(self.steering_angle) < return_change:
                    self.steering_angle = 0.0
                else:
                    self.steering_angle -= math.copysign(return_change, self.steering_angle)
        
        # Update velocity based on throttle input
        if self.throttle_input > 0:
            # Accelerating
            self.velocity += self.config.acceleration * self.throttle_input * dt
        elif self.throttle_input < 0:
            # Braking
            self.velocity += self.config.braking * self.throttle_input * dt
        else:
            # Natural friction deceleration
            if abs(self.velocity) > 0.1:
                friction_decel = self.config.friction * dt
                if abs(self.velocity) < friction_decel:
                    self.velocity = 0.0
                else:
                    self.velocity -= math.copysign(friction_decel, self.velocity)
        
        # Clamp velocity
        self.velocity = clamp(self.velocity, -self.config.max_velocity * 0.3, self.config.max_velocity)
        
        # Kinematic Bicycle Model equations
        if abs(self.velocity) > 0.1:
            # Angular velocity: θ' = (v / L) * tan(δ)
            angular_velocity = (self.velocity / self.config.wheelbase) * math.tan(self.steering_angle)
            
            # Update heading
            self.angle += angular_velocity * dt
            self.angle = normalize_angle(self.angle)
        
        # Position update: x' = v * cos(θ), y' = v * sin(θ)
        self.x += self.velocity * math.cos(self.angle) * dt
        self.y += self.velocity * math.sin(self.angle) * dt
    
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
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.throttle_input = 0.0
        self.steering_input = 0.0
    
    def get_corners(self) -> list[tuple[float, float]]:
        """Get the four corners of the car in world coordinates."""
        half_length = self.config.length / 2
        half_width = self.config.width / 2
        
        # Local corners (car center at origin, facing right)
        corners_local = [
            (half_length, -half_width),   # Front right
            (half_length, half_width),    # Front left
            (-half_length, half_width),   # Rear left
            (-half_length, -half_width),  # Rear right
        ]
        
        # Rotate and translate to world coordinates
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        corners_world = []
        for lx, ly in corners_local:
            wx = self.x + lx * cos_a - ly * sin_a
            wy = self.y + lx * sin_a + ly * cos_a
            corners_world.append((wx, wy))
        
        return corners_world
    
    def render(self, surface: pygame.Surface, debug: bool = False) -> None:
        """
        Render the car to a Pygame surface.
        
        Args:
            surface: Pygame surface to draw on
            debug: If True, draw additional debug info
        """
        corners = self.get_corners()
        
        # Draw car body
        pygame.draw.polygon(surface, self.color, corners)
        pygame.draw.polygon(surface, (255, 255, 255), corners, 2)  # White outline
        
        # Draw heading indicator (front of car)
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
            # Draw velocity vector
            if abs(self.velocity) > 1:
                vel_scale = 0.3
                vel_end = (
                    self.x + self.velocity * vel_scale * math.cos(self.angle),
                    self.y + self.velocity * vel_scale * math.sin(self.angle)
                )
                pygame.draw.line(surface, (0, 255, 0), (self.x, self.y), vel_end, 2)
            
            # Draw steering direction
            if abs(self.steering_angle) > 0.01:
                steer_angle = self.angle + self.steering_angle
                steer_end = (
                    front_center[0] + 25 * math.cos(steer_angle),
                    front_center[1] + 25 * math.sin(steer_angle)
                )
                pygame.draw.line(surface, (255, 100, 100), front_center, steer_end, 2)
    
    @property
    def speed_kmh(self) -> float:
        """Get speed in a human-readable format (arbitrary units resembling km/h)."""
        return abs(self.velocity) * 0.36  # Scale for display
    
    def get_state(self) -> dict:
        """Get current state as a dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'velocity': self.velocity,
            'steering_angle': self.steering_angle,
            'speed_kmh': self.speed_kmh
        }
