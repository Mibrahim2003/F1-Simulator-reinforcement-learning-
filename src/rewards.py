"""
Advanced Reward System for F1 Racing RL.

A state-of-the-art reward shaping system that enforces correct driving direction
using track progress tracking, centerline following, and directional checkpoints.
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Waypoint:
    """A waypoint on the racing line."""
    x: float
    y: float
    direction: float  # Expected heading at this point (radians)
    track_progress: float  # 0.0 to 1.0, position along track


@dataclass 
class RewardConfig:
    """Configuration for the reward system."""
    # Progress rewards (most important - enforces direction)
    progress_reward_scale: float = 10.0      # Reward per unit of forward progress
    wrong_way_penalty: float = -5.0          # Penalty per unit of backward progress
    
    # Speed rewards (scaled by correct direction)
    speed_reward_scale: float = 0.02         # Reward for speed when going correct way
    optimal_speed_bonus: float = 0.01        # Extra bonus near optimal speed
    
    # Centerline following
    centerline_reward_scale: float = 0.5     # Reward for staying near racing line
    max_centerline_distance: float = 75.0    # Max distance to track center
    
    # Heading alignment
    heading_alignment_scale: float = 1.0     # Reward for facing correct direction
    
    # Checkpoints (direction-enforced)
    checkpoint_bonus: float = 50.0           # Bonus for correct checkpoint order
    wrong_checkpoint_penalty: float = -25.0  # Penalty for wrong order (going backwards)
    lap_bonus: float = 500.0                 # Bonus for completing a lap
    
    # Penalties
    crash_penalty: float = -100.0            # Penalty for crashing
    time_penalty: float = -0.05              # Small per-step penalty
    stationary_penalty: float = -0.5         # Penalty for not moving
    reverse_penalty: float = -2.0            # Penalty for driving in reverse gear


class TrackProgressTracker:
    """
    Tracks car progress along the racing line using waypoints.
    This is the core system that enforces correct driving direction.
    """
    
    def __init__(self, track_width: int = 1280, track_height: int = 720,
                 external_waypoints: Optional[List] = None):
        self.track_width = track_width
        self.track_height = track_height
        self.waypoints: List[Waypoint] = []
        self.num_waypoints = 0
        
        if external_waypoints is not None:
            # Use waypoints from TrackLoader
            for wp in external_waypoints:
                self.waypoints.append(Waypoint(
                    x=wp['x'],
                    y=wp['y'],
                    direction=wp['direction'],
                    track_progress=wp['track_progress'],
                ))
            self.num_waypoints = len(self.waypoints)
        else:
            # Fall back to hardcoded oval waypoints
            self._generate_oval_waypoints()
    
    def _generate_oval_waypoints(self, num_points: int = 64):
        """
        Generate waypoints around the oval track.
        
        The oval track has:
        - Center at (640, 360)
        - Approximate racing line radius of 225 pixels from center
        - Counter-clockwise direction (car starts at bottom, goes up on the right)
        """
        cx, cy = self.track_width // 2, self.track_height // 2
        
        # Ellipse parameters matching track geometry
        # Track is an oval: wider horizontally than vertically
        radius_x = 425  # Horizontal radius of racing line
        radius_y = 225  # Vertical radius of racing line
        
        self.waypoints = []
        
        for i in range(num_points):
            # Angle around the track (counter-clockwise from bottom)
            # Start at bottom (270 degrees = 3*pi/2), go counter-clockwise
            t = (i / num_points) * 2 * math.pi
            angle = (3 * math.pi / 2) - t  # Counter-clockwise from bottom
            
            # Position on racing line
            x = cx + radius_x * math.cos(angle)
            y = cy + radius_y * math.sin(angle)
            
            # Expected heading (tangent to the ellipse, counter-clockwise)
            # Derivative of position gives tangent direction
            tangent_x = -radius_x * math.sin(angle) * (-1)  # Counter-clockwise
            tangent_y = radius_y * math.cos(angle) * (-1)
            heading = math.atan2(tangent_y, tangent_x)
            
            # Progress along track (0 to 1)
            progress = i / num_points
            
            self.waypoints.append(Waypoint(
                x=x,
                y=y,
                direction=heading,
                track_progress=progress
            ))
        
        self.num_waypoints = len(self.waypoints)
    
    def get_nearest_waypoint(self, x: float, y: float) -> Tuple[int, float]:
        """
        Find the nearest waypoint to a position.
        
        Returns:
            (waypoint_index, distance)
        """
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, wp in enumerate(self.waypoints):
            dist = math.sqrt((x - wp.x) ** 2 + (y - wp.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx, min_dist
    
    def get_track_progress(self, x: float, y: float) -> float:
        """
        Get the car's progress along the track (0 to 1).
        Uses interpolation between nearest waypoints for smoothness.
        """
        idx, dist = self.get_nearest_waypoint(x, y)
        return self.waypoints[idx].track_progress
    
    def get_expected_heading(self, x: float, y: float) -> float:
        """Get the expected heading at a position."""
        idx, _ = self.get_nearest_waypoint(x, y)
        return self.waypoints[idx].direction
    
    def get_distance_to_centerline(self, x: float, y: float) -> float:
        """Get distance from the racing line."""
        _, dist = self.get_nearest_waypoint(x, y)
        return dist
    
    def calculate_progress_delta(
        self, 
        prev_progress: float, 
        curr_progress: float
    ) -> float:
        """
        Calculate the progress made, handling wrap-around at lap completion.
        
        Returns:
            Positive value for forward progress, negative for backward.
        """
        delta = curr_progress - prev_progress
        
        # Handle wrap-around (crossing from 0.99 to 0.01 is forward progress)
        if delta < -0.5:
            delta += 1.0  # Wrapped forward
        elif delta > 0.5:
            delta -= 1.0  # Wrapped backward
        
        return delta


class AdvancedRewardCalculator:
    """
    State-of-the-art reward calculator for F1 Racing RL.
    
    Key features:
    1. Progress-based rewards (enforces correct direction)
    2. Centerline following (encourages optimal racing line)
    3. Heading alignment (rewards facing correct direction)
    4. Speed optimization (faster is better, but only going forward)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None,
                 external_waypoints: Optional[List] = None):
        self.config = config or RewardConfig()
        self.progress_tracker = TrackProgressTracker(
            external_waypoints=external_waypoints
        )
        
        # State tracking
        self.prev_progress: float = 0.0
        self.total_progress: float = 0.0
        self.laps_completed: int = 0
        self.last_checkpoint_idx: int = -1
        
        # Statistics for debugging
        self.reward_components: dict = {}
    
    def reset(self, start_x: float, start_y: float):
        """Reset the reward calculator for a new episode."""
        self.prev_progress = self.progress_tracker.get_track_progress(start_x, start_y)
        
        # If starting near end of track (grid position), assume Lap -1
        # This prevents immediate "Lap Complete" when crossing start line from grid
        if self.prev_progress > 0.8:
            self.laps_completed = -1
            self.total_progress = float(self.laps_completed) + self.prev_progress
        else:
            self.laps_completed = 0
            self.total_progress = self.prev_progress
            
        self.last_checkpoint_idx = -1
        self.reward_components = {}
    
    def calculate_reward(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        car_velocity: float,
        crashed: bool,
        checkpoint_crossed: Optional[int] = None,
        max_velocity: float = 300.0
    ) -> Tuple[float, dict]:
        """
        Calculate the total reward for the current step.
        
        Returns:
            (total_reward, reward_components_dict)
        """
        self.reward_components = {}
        total_reward = 0.0
        
        # 1. Crash penalty (terminal)
        if crashed:
            self.reward_components['crash'] = self.config.crash_penalty
            return self.config.crash_penalty, self.reward_components
        
        # 2. Progress reward (MOST IMPORTANT - enforces direction)
        curr_progress = self.progress_tracker.get_track_progress(car_x, car_y)
        progress_delta = self.progress_tracker.calculate_progress_delta(
            self.prev_progress, curr_progress
        )
        
        if progress_delta > 0:
            # Forward progress - reward!
            progress_reward = progress_delta * self.config.progress_reward_scale * 100
            self.reward_components['forward_progress'] = progress_reward
            total_reward += progress_reward
            
            # Check for lap completion (Crossing start line forward)
            # progress_delta was adjusted for wrap around. 
            # If we wrapped, the raw diff (curr - prev) would be ~ -0.9
            if (curr_progress - self.prev_progress) < -0.5:
                 self.laps_completed += 1
                 self.reward_components['lap_bonus'] = self.config.lap_bonus
                 total_reward += self.config.lap_bonus

        else:
            # Backward progress - penalty!
            wrong_way_penalty = progress_delta * abs(self.config.wrong_way_penalty) * 100
            self.reward_components['wrong_way'] = wrong_way_penalty
            total_reward += wrong_way_penalty
            
            # Check for lap decrement (Crossing start line backward)
            if (curr_progress - self.prev_progress) > 0.5:
                self.laps_completed -= 1

        # Total progress is Laps + Current Position (Absolute)
        # We don't accumulate delta for total anymore, we use absolute
        self.total_progress = float(self.laps_completed) + curr_progress
        self.prev_progress = curr_progress
        
        # 3. Speed reward (only when going in correct direction)
        if progress_delta > 0 and car_velocity > 0:
            normalized_speed = car_velocity / max_velocity
            speed_reward = normalized_speed * self.config.speed_reward_scale
            self.reward_components['speed'] = speed_reward
            total_reward += speed_reward
        
        # 4. Reverse gear penalty
        if car_velocity < 0:
            self.reward_components['reverse'] = self.config.reverse_penalty
            total_reward += self.config.reverse_penalty
        
        # 5. Stationary penalty
        if abs(car_velocity) < 5.0:
            self.reward_components['stationary'] = self.config.stationary_penalty
            total_reward += self.config.stationary_penalty
        
        # 6. Heading alignment reward
        expected_heading = self.progress_tracker.get_expected_heading(car_x, car_y)
        heading_diff = self._normalize_angle(car_angle - expected_heading)
        heading_alignment = math.cos(heading_diff)  # 1 when aligned, -1 when opposite
        
        heading_reward = heading_alignment * self.config.heading_alignment_scale
        self.reward_components['heading'] = heading_reward
        total_reward += heading_reward
        
        # 7. Centerline following reward
        centerline_dist = self.progress_tracker.get_distance_to_centerline(car_x, car_y)
        centerline_score = max(0, 1 - centerline_dist / self.config.max_centerline_distance)
        centerline_reward = centerline_score * self.config.centerline_reward_scale
        self.reward_components['centerline'] = centerline_reward
        total_reward += centerline_reward
        
        # 8. Checkpoint bonus (with order validation)
        if checkpoint_crossed is not None:
            expected_checkpoint = (self.last_checkpoint_idx + 1) % 4  # Assuming 4 checkpoints
            if checkpoint_crossed == expected_checkpoint:
                self.reward_components['checkpoint'] = self.config.checkpoint_bonus
                total_reward += self.config.checkpoint_bonus
                self.last_checkpoint_idx = checkpoint_crossed
            else:
                # Wrong order - might be going backwards
                self.reward_components['wrong_checkpoint'] = self.config.wrong_checkpoint_penalty
                total_reward += self.config.wrong_checkpoint_penalty
        
        # 9. Time penalty (encourages fast completion)
        self.reward_components['time'] = self.config.time_penalty
        total_reward += self.config.time_penalty
        
        return total_reward, self.reward_components
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_stats(self) -> dict:
        """Get current progress statistics."""
        return {
            'total_progress': self.total_progress,
            'laps_completed': self.laps_completed,
            'last_checkpoint': self.last_checkpoint_idx,
        }
