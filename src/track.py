"""
Track module for loading track images and handling collisions.
Uses Pygame masks for pixel-perfect collision detection.
"""
import pygame
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class Checkpoint:
    """A checkpoint line on the track for lap detection."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    index: int
    is_finish_line: bool = False


class Track:
    """
    Handles track loading, rendering, and collision detection.
    
    The track uses two images:
    - Visual track: The rendered appearance
    - Collision mask: Binary mask where white = wall, black = drivable
    """
    
    def __init__(self, track_path: str, mask_path: Optional[str] = None):
        """
        Load a track from image files.
        
        Args:
            track_path: Path to the visual track image
            mask_path: Path to the collision mask (defaults to track_path with _mask suffix)
        """
        # Derive mask path if not provided
        if mask_path is None:
            base, ext = os.path.splitext(track_path)
            mask_path = f"{base}_mask{ext}"
        
        # Load visual track
        self.image = pygame.image.load(track_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.width = self.rect.width
        self.height = self.rect.height
        
        # Load collision mask
        mask_image = pygame.image.load(mask_path).convert()
        self.collision_mask = pygame.mask.from_threshold(
            mask_image, 
            (255, 255, 255),  # White pixels = collision
            (128, 128, 128)   # Threshold tolerance
        )
        
        # Checkpoints for lap detection
        self.checkpoints: List[Checkpoint] = []
        self._setup_default_checkpoints()
        
        # Track statistics
        self.name = os.path.basename(track_path)
        
        # Track loader reference (None for image-based tracks)
        self._loader = None
    
    @classmethod
    def from_loader(cls, loader) -> 'Track':
        """
        Create a Track from a TrackLoader (CSV-based track).
        
        Args:
            loader: A TrackLoader instance with computed data
            
        Returns:
            Track instance with generated surfaces and checkpoints
        """
        track = cls.__new__(cls)
        
        # We don't generate surfaces anymore - we use loader directly
        # Set dummy dimensions to avoid errors (infinite world)
        track.width = 100000 
        track.height = 100000
        track.rect = pygame.Rect(0, 0, track.width, track.height)
        
        # No collision mask image
        track.image = None
        track.collision_mask = None
        
        track.name = loader.track_name
        track._loader = loader
        
        # Set up checkpoints from loader
        track.checkpoints = []
        checkpoint_data = loader.compute_checkpoints(4)
        for start, end, idx, is_finish in checkpoint_data:
            track.checkpoints.append(Checkpoint(start, end, idx, is_finish))
        
        return track
    
    def _setup_default_checkpoints(self):
        """Set up default checkpoints for the oval track."""
        cx, cy = self.width // 2, self.height // 2
        
        # Track geometry: outer_radius_y=300, track_width=150
        # Stroke center is at 300 - 75 = 225 from screen center
        track_center_offset = 225
        checkpoint_half_width = 75  # Half the track width
        
        # Simple checkpoint system for oval track
        checkpoints_data = [
            # Finish line at bottom
            ((cx - checkpoint_half_width, cy + track_center_offset), 
             (cx + checkpoint_half_width, cy + track_center_offset), 0, True),
            # Checkpoint at right
            ((cx + 425, cy - checkpoint_half_width), 
             (cx + 425, cy + checkpoint_half_width), 1, False),
            # Checkpoint at top
            ((cx - checkpoint_half_width, cy - track_center_offset), 
             (cx + checkpoint_half_width, cy - track_center_offset), 2, False),
            # Checkpoint at left
            ((cx - 425, cy - checkpoint_half_width), 
             (cx - 425, cy + checkpoint_half_width), 3, False),
        ]
        
        for start, end, idx, is_finish in checkpoints_data:
            self.checkpoints.append(Checkpoint(start, end, idx, is_finish))
    
    def check_collision(self, x: float, y: float) -> bool:
        """
        Check if a point collides with the track walls.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if the point is on a wall (collision), False if drivable
        """
        # Delegate to loader if available (math-based collision)
        if self._loader:
            # check_collision returns True if drivable, False if wall
            # This method returns True if collision (wall)
            return not self._loader.check_collision(x, y)
            
        # Legacy image-based check
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
            return True  # Out of bounds = collision
        
        # Check mask at this point
        return self.collision_mask.get_at((ix, iy)) == 1
    
    def check_car_collision(self, car_corners: List[Tuple[float, float]]) -> bool:
        """
        Check if any corner of the car collides with walls.
        
        Args:
            car_corners: List of (x, y) tuples for car corners
            
        Returns:
            True if any corner is on a wall
        """
        for x, y in car_corners:
            if self.check_collision(x, y):
                return True
        return False
    
    def raycast(self, start_x: float, start_y: float, angle: float, 
                max_distance: float = 500.0) -> Tuple[float, Tuple[int, int]]:
        """
        Cast a ray from a point in a direction until it hits a wall.
        
        Args:
            start_x: Starting X position
            start_y: Starting Y position
            angle: Ray direction in radians
            max_distance: Maximum ray length
            
        Returns:
            Tuple of (distance to wall, hit point (x, y))
        """
        # Delegate to loader if available
        if self._loader:
            return self._loader.raycast(start_x, start_y, angle, max_distance)
            
        import math
        
        # Ray marching with small steps for accuracy
        step_size = 3.0  # Pixels per step
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        distance = 0.0
        x, y = start_x, start_y
        
        while distance < max_distance:
            x = start_x + distance * cos_a
            y = start_y + distance * sin_a
            
            # Check bounds
            ix, iy = int(x), int(y)
            if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
                return distance, (ix, iy)
            
            # Check collision
            if self.collision_mask.get_at((ix, iy)) == 1:
                return distance, (ix, iy)
            
            distance += step_size
        
        # No collision within max distance
        return max_distance, (int(start_x + max_distance * cos_a), 
                              int(start_y + max_distance * sin_a))
    
    def check_checkpoint_crossing(self, prev_pos: Tuple[float, float], 
                                   curr_pos: Tuple[float, float],
                                   checkpoint: Checkpoint) -> bool:
        """
        Check if a car crossed a checkpoint line between two positions.
        Uses line segment intersection.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A, B = prev_pos, curr_pos
        C, D = checkpoint.start, checkpoint.end
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def render(self, surface: pygame.Surface, debug: bool = False):
        """Render the track to a surface."""
        surface.blit(self.image, (0, 0))
        
        if debug:
            # Draw checkpoints
            for cp in self.checkpoints:
                color = (255, 255, 0) if cp.is_finish_line else (100, 100, 255)
                pygame.draw.line(surface, color, cp.start, cp.end, 3)
    
    def get_spawn_position(self) -> Tuple[float, float, float]:
        """
        Get the default spawn position and angle for a car.
        
        Returns:
            Tuple of (x, y, angle_radians)
        """
        import math
        
        # CSV-based track: use first grid position from loader
        if self._loader is not None:
            grid = self._loader.compute_grid_positions(1)
            if grid:
                return grid[0].x, grid[0].y, grid[0].angle
        
        # Image-based track: hardcoded oval spawn
        cx, cy = self.width // 2, self.height // 2
        track_center_offset = 225 - 25
        spawn_y = cy + track_center_offset
        return cx, spawn_y, -math.pi / 2  # Facing up
