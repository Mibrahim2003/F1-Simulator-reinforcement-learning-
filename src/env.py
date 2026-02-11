"""
F1 Racing Gymnasium Environment - Version 2.

Enhanced with state-of-the-art reward shaping that enforces correct driving direction
using track progress tracking and waypoint-based rewards.
"""
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from src.car import Car, CarConfig
from src.track import Track
from src.sensors import LidarSensor, SensorConfig
from src.rewards import AdvancedRewardCalculator, RewardConfig


class F1RacingEnv(gym.Env):
    """
    F1 Racing Environment for Reinforcement Learning - V2.
    
    Enhanced with direction-aware reward system that prevents backwards driving.
    
    Observation Space:
        - LIDAR distances (normalized 0-1): num_rays values
        - Velocity (normalized): 1 value
        - Steering angle (normalized): 1 value
        - Track progress (0-1): 1 value (NEW)
        - Heading alignment (-1 to 1): 1 value (NEW)
        Total: num_rays + 4 values
    
    Action Space (Continuous):
        - [0]: Throttle/Brake (-1 to 1)
        - [1]: Steering (-1 to 1)
    
    Rewards (NEW - Direction Enforced):
        - Progress reward: Only for moving in CORRECT direction
        - Wrong-way penalty: For moving backwards along track
        - Speed bonus: Only when making forward progress
        - Heading alignment: Reward for facing correct direction
        - Centerline following: Reward for optimal racing line
        - Checkpoint bonuses: Order-validated
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        track_path: str = "assets/tracks/oval_track.png",
        num_lidar_rays: int = 11,
        lidar_fov: float = math.pi,
        max_episode_steps: int = 2000,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Initialize the F1 Racing environment.
        
        Args:
            render_mode: 'human' for window display, 'rgb_array' for pixel data
            track_path: Path to track image file
            num_lidar_rays: Number of LIDAR sensor rays
            lidar_fov: Field of view for LIDAR in radians
            max_episode_steps: Maximum steps before truncation
            reward_config: Configuration for the reward system
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.track_path = track_path
        self.max_episode_steps = max_episode_steps
        
        # Initialize reward calculator
        self.reward_config = reward_config or RewardConfig()
        self.reward_calculator = AdvancedRewardCalculator(self.reward_config)
        
        # Sensor configuration
        self.num_lidar_rays = num_lidar_rays
        self.sensor_config = SensorConfig(
            num_rays=num_lidar_rays,
            fov=lidar_fov,
            max_distance=350.0
        )
        
        # Define observation space
        # [LIDAR (num_rays), velocity (1), steering (1), progress (1), heading_align (1)]
        obs_dim = num_lidar_rays + 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space (continuous)
        # [throttle/brake, steering]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Will be initialized on first reset
        self.track: Optional[Track] = None
        self.car: Optional[Car] = None
        self.sensor: Optional[LidarSensor] = None
        
        # Episode state
        self.steps = 0
        self.total_reward = 0.0
        self.prev_pos = (0, 0)
        
        # Rendering
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self._pygame_initialized = False
        
        # Debug: track reward components
        self.last_reward_components: dict = {}
    
    def _init_pygame(self):
        """Initialize Pygame for rendering."""
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("F1 RL Racing - V2 (Direction Enforced)")
                self.screen = pygame.display.set_mode((1280, 720))
            else:
                pygame.display.set_mode((1, 1), pygame.HIDDEN)
                self.screen = pygame.Surface((1280, 720))
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont("Consolas", 14)
            except:
                self.font = pygame.font.Font(None, 18)
            self._pygame_initialized = True
    
    def _init_simulation(self):
        """Initialize track, car, and sensors."""
        self._init_pygame()
        self.track = Track(self.track_path)
        
        spawn_x, spawn_y, spawn_angle = self.track.get_spawn_position()
        self.car = Car(x=spawn_x, y=spawn_y, angle=spawn_angle, config=CarConfig())
        self.sensor = LidarSensor(self.sensor_config)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector with enhanced state info."""
        # LIDAR readings
        lidar_readings = self.sensor.update(
            self.car.x, self.car.y, self.car.angle, self.track
        )
        
        # Normalized velocity [-1, 1]
        normalized_velocity = self.car.velocity / self.car.config.max_velocity
        
        # Normalized steering [-1, 1]
        normalized_steering = self.car.steering_angle / self.car.config.max_steering_angle
        
        # Track progress [0, 1] - NEW
        progress = self.reward_calculator.progress_tracker.get_track_progress(
            self.car.x, self.car.y
        )
        # Scale to [-1, 1] for consistency
        normalized_progress = progress * 2 - 1
        
        # Heading alignment [-1, 1] - NEW
        expected_heading = self.reward_calculator.progress_tracker.get_expected_heading(
            self.car.x, self.car.y
        )
        heading_diff = self._normalize_angle(self.car.angle - expected_heading)
        heading_alignment = math.cos(heading_diff)  # 1 = correct, -1 = backwards
        
        obs = np.array(
            lidar_readings + [
                normalized_velocity,
                normalized_steering,
                normalized_progress,
                heading_alignment
            ],
            dtype=np.float32
        )
        
        return obs
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if self.track is None:
            self._init_simulation()
        
        # Reset car
        spawn_x, spawn_y, spawn_angle = self.track.get_spawn_position()
        self.car.reset(spawn_x, spawn_y, spawn_angle)
        
        # Reset reward calculator
        self.reward_calculator.reset(spawn_x, spawn_y)
        
        # Reset episode state
        self.steps = 0
        self.total_reward = 0.0
        self.prev_pos = (self.car.x, self.car.y)
        self.last_reward_components = {}
        
        observation = self._get_observation()
        
        info = {
            "laps": 0,
            "progress": 0.0,
            "velocity": 0.0,
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with enhanced rewards."""
        self.steps += 1
        
        # Apply action
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))
        self.car.set_controls(throttle, steering)
        
        # Update physics
        dt = 1.0 / 60.0
        self.car.update(dt)
        
        # Check collision
        crashed = self.track.check_car_collision(self.car.get_corners())
        
        # Check checkpoint crossing
        checkpoint_crossed = None
        curr_pos = (self.car.x, self.car.y)
        for checkpoint in self.track.checkpoints:
            if self.track.check_checkpoint_crossing(self.prev_pos, curr_pos, checkpoint):
                checkpoint_crossed = checkpoint.index
                break
        self.prev_pos = curr_pos
        
        # Calculate reward using advanced system
        reward, components = self.reward_calculator.calculate_reward(
            car_x=self.car.x,
            car_y=self.car.y,
            car_angle=self.car.angle,
            car_velocity=self.car.velocity,
            crashed=crashed,
            checkpoint_crossed=checkpoint_crossed,
            max_velocity=self.car.config.max_velocity
        )
        
        self.last_reward_components = components
        self.total_reward += reward
        
        # Termination
        terminated = crashed
        truncated = self.steps >= self.max_episode_steps
        
        # Observation
        observation = self._get_observation()
        
        # Info
        stats = self.reward_calculator.get_stats()
        info = {
            "laps": stats['laps_completed'],
            "progress": stats['total_progress'],
            "velocity": self.car.velocity,
            "total_reward": self.total_reward,
            "crashed": crashed,
            "reward_components": components,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment with enhanced HUD."""
        if self.render_mode is None:
            return None
        
        self._init_pygame()
        
        if self.track is None:
            return None
        
        self.screen.fill((30, 30, 40))
        
        # Draw track
        self.track.render(self.screen, debug=True)
        
        # Draw waypoints (debug) - show racing line
        self._draw_racing_line()
        
        # Draw LIDAR
        self.sensor.render(self.screen, self.car.x, self.car.y)
        
        # Draw car
        self.car.render(self.screen, debug=False)
        
        # Draw enhanced HUD
        self._render_hud()
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )
    
    def _draw_racing_line(self):
        """Draw the racing line (waypoints) for debugging."""
        tracker = self.reward_calculator.progress_tracker
        
        # Draw waypoints as small dots
        for i, wp in enumerate(tracker.waypoints):
            # Color based on position (gradient around track)
            hue = i / len(tracker.waypoints)
            color = self._hsv_to_rgb(hue, 0.7, 0.8)
            pygame.draw.circle(self.screen, color, (int(wp.x), int(wp.y)), 3)
        
        # Highlight nearest waypoint
        idx, _ = tracker.get_nearest_waypoint(self.car.x, self.car.y)
        wp = tracker.waypoints[idx]
        pygame.draw.circle(self.screen, (255, 255, 0), (int(wp.x), int(wp.y)), 6, 2)
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _render_hud(self):
        """Render enhanced heads-up display with reward breakdown."""
        # Main panel
        panel_rect = pygame.Rect(10, 10, 220, 160)
        pygame.draw.rect(self.screen, (20, 20, 30, 220), panel_rect, border_radius=5)
        pygame.draw.rect(self.screen, (0, 200, 150), panel_rect, 2, border_radius=5)
        
        stats = self.reward_calculator.get_stats()
        
        lines = [
            f"Step: {self.steps}/{self.max_episode_steps}",
            f"Speed: {self.car.speed_kmh:.1f} km/h",
            f"Progress: {stats['total_progress']*100:.1f}%",
            f"Laps: {stats['laps_completed']}",
            f"Total Reward: {self.total_reward:.1f}",
            "",
            "Reward Breakdown:",
        ]
        
        # Add reward components
        for key, value in self.last_reward_components.items():
            if abs(value) > 0.001:
                sign = "+" if value > 0 else ""
                lines.append(f"  {key}: {sign}{value:.2f}")
        
        y = 15
        for line in lines:
            if line.startswith("  "):
                color = (150, 255, 150) if "+" in line else (255, 150, 150)
            else:
                color = (220, 220, 220)
            text = self.font.render(line, True, color)
            self.screen.blit(text, (15, y))
            y += 18
    
    def close(self):
        """Clean up resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self.screen = None
            self.clock = None


# Register the environment
def register_env():
    """Register the F1 Racing environment with Gymnasium."""
    gym.register(
        id="F1Racing-v1",
        entry_point="src.env:F1RacingEnv",
        max_episode_steps=2000,
    )
