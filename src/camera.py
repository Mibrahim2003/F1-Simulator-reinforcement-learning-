
import pygame
import math
from typing import Tuple, Optional

class Camera:
    """
    A 2D camera that follows a target and handles coordinate transformations
    with zoom support.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Camera center point on screen
        self.center_x = width // 2
        self.center_y = height // 2
        
        # World position of the camera center
        self.x = 0.0
        self.y = 0.0
        
        # Zoom level (1.0 = normal, >1.0 = zoom in, <1.0 = zoom out)
        self.zoom = 1.0
        self.target_zoom = 1.0
        
        # Camera movement smoothing (0.0 = instant, 1.0 = no movement)
        self.smooth_speed = 0.1
        self.zoom_speed = 0.05
        
    def update(self, target_pos: Tuple[float, float]):
        """
        Update camera position to follow target smoothly.
        """
        tx, ty = target_pos
        
        # Smooth follow
        self.x += (tx - self.x) * self.smooth_speed
        self.y += (ty - self.y) * self.smooth_speed
        
        # Smooth zoom
        self.zoom += (self.target_zoom - self.zoom) * self.zoom_speed
        self.zoom = max(0.1, min(self.zoom, 5.0))  # Clamp zoom
        
    def set_zoom(self, zoom: float):
        """Set target zoom level."""
        self.target_zoom = max(0.1, zoom)
        
    def transform(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        """
        # Translate to camera center, scale, then translate to screen center
        dx = (x - self.x) * self.zoom
        dy = (y - self.y) * self.zoom
        
        return (int(self.center_x + dx), int(self.center_y + dy))
        
    def inverse_transform(self, sx: int, sy: int) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.
        """
        # Determine offset from center, unscale, then add camera position
        dx = (sx - self.center_x) / self.zoom
        dy = (sy - self.center_y) / self.zoom
        
        return (self.x + dx, self.y + dy)
        
    def get_visible_rect(self) -> pygame.Rect:
        """
        Get the rectangle of the world currently visible on screen.
        """
        # Top-left in world coords
        x1, y1 = self.inverse_transform(0, 0)
        # Bottom-right in world coords
        x2, y2 = self.inverse_transform(self.width, self.height)
        
        return pygame.Rect(x1, y1, x2 - x1, y2 - y1)
