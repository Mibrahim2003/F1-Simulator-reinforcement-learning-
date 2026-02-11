"""
LIDAR Sensor system for the car.
Casts multiple rays to detect distances to walls, providing observations for RL.
"""
import math
import pygame
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SensorConfig:
    """Configuration for the LIDAR sensor."""
    num_rays: int = 9           # Number of rays to cast
    fov: float = math.pi        # Field of view in radians (180 degrees)
    max_distance: float = 400.0  # Maximum sensing distance
    
    # Visualization
    ray_width: int = 2
    show_hit_points: bool = True


class LidarSensor:
    """
    LIDAR sensor that casts rays from the car to detect walls.
    
    Provides a 1D array of distances that can be used as observations
    for reinforcement learning.
    """
    
    def __init__(self, config: SensorConfig = None):
        """
        Initialize the LIDAR sensor.
        
        Args:
            config: Sensor configuration
        """
        self.config = config or SensorConfig()
        
        # Calculate ray angles (evenly distributed across FOV)
        self.ray_angles = self._calculate_ray_angles()
        
        # Last readings (distances and hit points)
        self.distances: List[float] = [self.config.max_distance] * self.config.num_rays
        self.hit_points: List[Tuple[int, int]] = [(0, 0)] * self.config.num_rays
        self.normalized_distances: List[float] = [1.0] * self.config.num_rays
    
    def _calculate_ray_angles(self) -> List[float]:
        """Calculate the angle offset for each ray relative to car heading."""
        if self.config.num_rays == 1:
            return [0.0]
        
        # Distribute rays evenly across FOV, centered on forward direction
        half_fov = self.config.fov / 2
        angles = []
        
        for i in range(self.config.num_rays):
            # -half_fov to +half_fov
            ratio = i / (self.config.num_rays - 1)  # 0 to 1
            angle = -half_fov + ratio * self.config.fov
            angles.append(angle)
        
        return angles
    
    def update(self, car_x: float, car_y: float, car_angle: float, 
               track: 'Track') -> List[float]:
        """
        Update sensor readings by casting rays.
        
        Args:
            car_x: Car center X position
            car_y: Car center Y position
            car_angle: Car heading angle in radians
            track: Track object with raycast method
            
        Returns:
            List of normalized distances (0 = wall at car, 1 = max distance)
        """
        self.distances = []
        self.hit_points = []
        self.normalized_distances = []
        
        for ray_offset in self.ray_angles:
            # Calculate world angle for this ray
            world_angle = car_angle + ray_offset
            
            # Cast the ray
            distance, hit_point = track.raycast(
                car_x, car_y, 
                world_angle, 
                self.config.max_distance
            )
            
            self.distances.append(distance)
            self.hit_points.append(hit_point)
            self.normalized_distances.append(distance / self.config.max_distance)
        
        return self.normalized_distances
    
    def get_observation(self) -> List[float]:
        """Get normalized distances as observation for RL."""
        return self.normalized_distances.copy()
    
    def render(self, surface: pygame.Surface, car_x: float, car_y: float):
        """
        Render the LIDAR rays on the screen.
        
        Args:
            surface: Pygame surface to draw on
            car_x: Car center X position
            car_y: Car center Y position
        """
        for i, (distance, hit_point) in enumerate(zip(self.distances, self.hit_points)):
            # Color based on distance (red = close, green = far)
            ratio = distance / self.config.max_distance
            
            # Interpolate from red (close) to green (far)
            red = int(255 * (1 - ratio))
            green = int(255 * ratio)
            color = (red, green, 50)
            
            # Draw ray line
            pygame.draw.line(
                surface, 
                color, 
                (int(car_x), int(car_y)), 
                hit_point,
                self.config.ray_width
            )
            
            # Draw hit point marker
            if self.config.show_hit_points:
                pygame.draw.circle(surface, (255, 255, 255), hit_point, 4)
                pygame.draw.circle(surface, color, hit_point, 3)
    
    def render_minimap(self, surface: pygame.Surface, center: Tuple[int, int], 
                       scale: float = 0.3):
        """
        Render a mini radar-style view of the sensor readings.
        
        Args:
            surface: Pygame surface
            center: Center point of the minimap
            scale: Scale factor for distances
        """
        # Draw background circle
        max_radius = int(self.config.max_distance * scale)
        pygame.draw.circle(surface, (30, 30, 40), center, max_radius + 5)
        pygame.draw.circle(surface, (60, 60, 70), center, max_radius + 5, 2)
        
        # Draw distance rings
        for ring in [0.25, 0.5, 0.75, 1.0]:
            radius = int(max_radius * ring)
            pygame.draw.circle(surface, (50, 50, 60), center, radius, 1)
        
        # Draw rays
        for i, (angle_offset, distance) in enumerate(zip(self.ray_angles, self.distances)):
            # In minimap, forward is up (-π/2), so adjust
            display_angle = -math.pi / 2 + angle_offset
            
            scaled_distance = distance * scale
            end_x = center[0] + scaled_distance * math.cos(display_angle)
            end_y = center[1] + scaled_distance * math.sin(display_angle)
            
            ratio = distance / self.config.max_distance
            red = int(255 * (1 - ratio))
            green = int(255 * ratio)
            color = (red, green, 50)
            
            pygame.draw.line(surface, color, center, (int(end_x), int(end_y)), 2)
            pygame.draw.circle(surface, (255, 255, 255), (int(end_x), int(end_y)), 3)
