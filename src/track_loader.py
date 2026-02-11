"""
Track Loader for CSV-based track data.

Loads real F1 track data from the TUMFTM racetrack database format:
  [x_m, y_m, w_tr_right_m, w_tr_left_m]

Handles scaling, boundary computation, rendering, and auto-generation of
waypoints, grid positions, and checkpoints.
"""
import csv
import math
import os
import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TrackPoint:
    """A single point on the track center line with width info."""
    x: float  # Center line x (meters, original)
    y: float  # Center line y (meters, original)
    w_right: float  # Track width to the right (meters)
    w_left: float   # Track width to the left (meters)


@dataclass
class ScaledPoint:
    """A point in pixel coordinates after scaling."""
    x: float
    y: float


@dataclass
class GridSlot:
    """A starting grid position."""
    x: float
    y: float
    angle: float  # Facing direction in radians


class TrackLoader:
    """
    Loads and processes track data from CSV files.
    
    Provides:
    - Scaled center line points (pixels)
    - Inner/outer boundary polygons
    - Rendered track surface and collision mask
    - Waypoints with headings for progress tracking
    - F1-style starting grid positions
    - Checkpoint lines for lap detection
    """
    
    def __init__(
        self,
        track_name: str,
        tracks_dir: str = "tracks_data/tracks",
        racelines_dir: str = "tracks_data/racelines",
    ):
        """
        Load a track from CSV data with realistic scaling.
        
        Args:
            track_name: Name of the track (e.g. "Monza", "Silverstone")
            tracks_dir: Path to the tracks CSV directory
            racelines_dir: Path to the racelines CSV directory
        """
        self.track_name = track_name
        self.PIXELS_PER_METER = 8.0  # constant fixed scale
        self.padding = 100  # Margin in world pixels
        
        # Raw data
        self.raw_points: List[TrackPoint] = []
        
        # Scaled data (pixels)
        self.center_line: List[ScaledPoint] = []
        self.inner_boundary: List[ScaledPoint] = []
        self.outer_boundary: List[ScaledPoint] = []
        self.racing_line: List[ScaledPoint] = []
        
        # Computed data
        self.headings: List[float] = []  # Direction at each center point
        self.grid_positions: List[GridSlot] = []
        self.checkpoint_lines: List[Tuple[Tuple[float, float], Tuple[float, float], int, bool]] = []
        
        # Scaling transform parameters
        self.scale = self.PIXELS_PER_METER
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        # Surfaces (generated on demand)
        # self._track_surface removed - we render dynamically
        # self._collision_mask removed - we use math collision
        
        # Load and process
        track_csv = os.path.join(tracks_dir, f"{track_name}.csv")
        raceline_csv = os.path.join(racelines_dir, f"{track_name}.csv")
        
        self._load_csv(track_csv)
        self._compute_scaling()
        self._compute_scaled_points()
        self._compute_headings()
        self._compute_boundaries()
        
        # Load racing line if available
        if os.path.exists(raceline_csv):
            self._load_racing_line(raceline_csv)
    
    def _load_csv(self, path: str):
        """Load center line + widths from CSV."""
        self.raw_points = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    self.raw_points.append(TrackPoint(
                        x=float(row[0]),
                        y=float(row[1]),
                        w_right=float(row[2]),
                        w_left=float(row[3]),
                    ))
    
    def _load_racing_line(self, path: str):
        """Load optimized racing line from CSV."""
        self.racing_line = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2:
                    px, py = self._meters_to_pixels(float(row[0]), float(row[1]))
                    self.racing_line.append(ScaledPoint(x=px, y=py))
    
    def _compute_scaling(self):
        """Compute the offset to place track in positive coordinates."""
        if not self.raw_points:
            return
        
        # Use fixed scale
        self.scale = self.PIXELS_PER_METER
        
        xs = [p.x for p in self.raw_points]
        ys = [p.y for p in self.raw_points]
        
        # Include boundary extents (add max track width)
        max_width = max(max(p.w_right for p in self.raw_points),
                       max(p.w_left for p in self.raw_points))
        
        data_min_x = min(xs) - max_width
        data_min_y = min(ys) - max_width
        
        # Offset to start at padding
        # We assume original X/Y can be negative, so we shift them
        self.offset_x = -data_min_x * self.scale + self.padding
        self.offset_y = -data_min_y * self.scale + self.padding
    def _meters_to_pixels(self, mx: float, my: float) -> Tuple[float, float]:
        """Convert meters to pixel coordinates."""
        px = mx * self.scale + self.offset_x
        # We don't flip Y anymore because world space is huge.
        # But wait, original data Y usually goes Up. Screen goes Down.
        # Yes, keep Y flip but relative to world space.
        # Check: _compute_scaling sets offset_y = -min_y * scale + padding.
        # So min_y (which is lowest Y value) maps to padding.
        # That means small Y -> small screen Y.
        # If original data has Y Up, then small Y is bottom.
        # We want small Y to be top? Or keep standard cartesian?
        # F1 track data is likely Cartesian (Y=North).
        # Screen Y=Down.
        # Let's flip Y: py = -my * scale + offset_y.
        # But update offset_y to be correct.
        
        # Logic:
        # We want max_y to be at top (small screen Y)? No, map view usually Y=North=Up.
        # In Pygame, Y=0 is Top.
        # So we want Max Y (North) to be at Y=0.
        # So py = (max_y - my) * scale + padding.
        # Let's re-read _compute_scaling above.
        # I set offset_y = -min_y * scale.
        # py = my * scale + offset. This maps min_y to 0. (Y=South is Top).
        # This means Y axis is inverted relative to standard map.
        # Let's use:
        # py = -my * scale + OFFSET
        pass 
        
        px = mx * self.scale + self.offset_x
        # Standard 2D game coords: Y is down.
        # Track data: Y is likely North.
        # Let's flip it so North is Up (Negative Y in pygame).
        # But we need to shift it so it's positive.
        # This is tricky without knowing data bounds here.
        # Actually _compute_scaling can't perfectly know bounds for flip.
        # Let's just use direct mapping x->x, y->y for now.
        # Because "Up" on screen is just a convention.
        py = my * self.scale + self.offset_y
        return px, py
    
    def _compute_scaled_points(self):
        """Scale all center line points to pixel coordinates."""
        self.center_line = []
        for p in self.raw_points:
            px, py = self._meters_to_pixels(p.x, p.y)
            self.center_line.append(ScaledPoint(x=px, y=py))
    
    def _compute_headings(self):
        """Compute heading (direction) at each center line point."""
        self.headings = []
        n = len(self.center_line)
        
        for i in range(n):
            # Use next point to compute forward direction
            next_i = (i + 1) % n
            dx = self.center_line[next_i].x - self.center_line[i].x
            dy = self.center_line[next_i].y - self.center_line[i].y
            heading = math.atan2(dy, dx)
            self.headings.append(heading)
    
    def _compute_boundaries(self):
        """Compute inner and outer track boundaries from center line + widths."""
        self.inner_boundary = []
        self.outer_boundary = []
        n = len(self.center_line)
        
        # Calculate width multiplier not needed with realistic scale
        self.width_multiplier = 1.0
        
        for i in range(n):
            heading = self.headings[i]
            
            # Perpendicular direction (90° to the right of heading)
            perp_x = -math.sin(heading)
            perp_y = math.cos(heading)
            
            # Scale widths from meters to pixels, with multiplier for gameplay
            w_right = self.raw_points[i].w_right * self.scale * self.width_multiplier
            w_left = self.raw_points[i].w_left * self.scale * self.width_multiplier
            
            cx = self.center_line[i].x
            cy = self.center_line[i].y
            
            # Right boundary (outer in most cases)
            self.outer_boundary.append(ScaledPoint(
                x=cx + perp_x * w_right,
                y=cy + perp_y * w_right,
            ))
            
            # Left boundary (inner in most cases)
            self.inner_boundary.append(ScaledPoint(
                x=cx - perp_x * w_left,
                y=cy - perp_y * w_left,
            ))
    
    def render(self, surface: pygame.Surface, camera):
        """
        Render the visible portion of the track.
        """
        # Grass background is handled by main loop clear, or we can draw it here
        # But for infinite world, clearing screen is better.
        
        n = len(self.center_line)
        if n < 2:
            return
            
        outer_pts = [(p.x, p.y) for p in self.outer_boundary]
        inner_pts = [(p.x, p.y) for p in self.inner_boundary]
        
        cam_rect = camera.get_visible_rect()
        # Inflate slightly to avoid popping
        cam_rect.inflate_ip(100, 100)
        
        # Function to transform point
        def t(x, y):
            return camera.transform(x, y)
        
        # Iterate segments
        for i in range(n):
            next_i = (i + 1) % n
            
            # Simple culling: check if bounding box of quad intersects camera
            xs = [outer_pts[i][0], outer_pts[next_i][0], inner_pts[next_i][0], inner_pts[i][0]]
            ys = [outer_pts[i][1], outer_pts[next_i][1], inner_pts[next_i][1], inner_pts[i][1]]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            if not cam_rect.colliderect(pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)):
                continue
            
            # Draw Quad
            quad = [
                t(outer_pts[i][0], outer_pts[i][1]),
                t(outer_pts[next_i][0], outer_pts[next_i][1]),
                t(inner_pts[next_i][0], inner_pts[next_i][1]),
                t(inner_pts[i][0], inner_pts[i][1])
            ]
            
            pygame.draw.polygon(surface, (80, 84, 92), quad)
            
            # Draw edges
            pygame.draw.line(surface, (200, 200, 210), quad[0], quad[1], 2)
            pygame.draw.line(surface, (200, 200, 210), quad[3], quad[2], 2)
            
            # Center line
            if i % 2 == 0: # Dashed
                 p1 = t(self.center_line[i].x, self.center_line[i].y)
                 p2 = t(self.center_line[next_i].x, self.center_line[next_i].y)
                 pygame.draw.line(surface, (100, 100, 110), p1, p2, 1)
        
        # Start line
        if self.center_line:
            p = self.center_line[0]
            # ... calculation is a bit complex for transform, simpler to just use outer/inner[0]
            # Start line is between inner[0] and outer[0]
            s = t(inner_pts[0][0], inner_pts[0][1])
            e = t(outer_pts[0][0], outer_pts[0][1])
            pygame.draw.line(surface, (255, 255, 255), s, e, 3)

    def check_collision(self, x: float, y: float) -> bool:
        """
        Check if point (x, y) is ON THE TRACK.
        Returns True if drivable, False if wall.
        """
        # Optimization: Find nearest center line point
        # Since points are ordered, we can just search.
        # For 1000 points, simple linear search is 0.05ms.
        # We can optimize by tracking current segment index from previous frame if needed.
        
        # 1. Find nearest centerline point index
        # (This is a simplified approach. Ideally we use spatial hash)
        min_dist_sq = float('inf')
        nearest_idx = -1
        
        # Optimization: only check points?
        # Let's brute force for now.
        for i, p in enumerate(self.center_line):
            d2 = (p.x - x)**2 + (p.y - y)**2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                nearest_idx = i
                
        # If too far from center line, it's definitely wall
        # Max track width is usually < 20m = 160px.
        if min_dist_sq > 200**2:
            return False # Wall
            
        # 2. Check precise collision with the quad of this segment (and neighbors)
        # We check idx, idx-1, idx+1 to be sure
        indices = [nearest_idx, (nearest_idx - 1) % len(self.center_line), (nearest_idx + 1) % len(self.center_line)]
        
        for i in indices:
            next_i = (i + 1) % len(self.center_line)
            # Quad: Outer[i], Outer[next], Inner[next], Inner[i]
            # Check point in polygon
            poly = [
                (self.outer_boundary[i].x, self.outer_boundary[i].y),
                (self.outer_boundary[next_i].x, self.outer_boundary[next_i].y),
                (self.inner_boundary[next_i].x, self.inner_boundary[next_i].y),
                (self.inner_boundary[i].x, self.inner_boundary[i].y)
            ]
            if self._point_in_polygon(x, y, poly):
                return True # Drivable
        
        return False # Wall

    def _point_in_polygon(self, x, y, poly):
        """Ray casting algorithm for point in polygon."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def raycast(self, start_x: float, start_y: float, angle: float, 
                max_distance: float = 500.0) -> Tuple[float, Tuple[int, int]]:
        """
        Cast a ray to find wall intersection using mathematical checks.
        Uses ray marching with binary search refinement.
        """
        step_size = 4.0 * self.scale # 4 meters step
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Check start point
        if not self.check_collision(start_x, start_y):
            # Started in wall
            return 0.0, (int(start_x), int(start_y))
            
        distance = 0.0
        x, y = start_x, start_y
        
        # Ray marching
        while distance < max_distance:
            next_dist = distance + step_size
            next_x = start_x + next_dist * cos_a
            next_y = start_y + next_dist * sin_a
            
            if not self.check_collision(next_x, next_y):
                # Hit wall between distance and next_dist
                # Binary search to refine
                low = distance
                high = next_dist
                
                for _ in range(4): # 4 iterations gives good enough precision
                    mid = (low + high) / 2
                    mid_x = start_x + mid * cos_a
                    mid_y = start_y + mid * sin_a
                    
                    if self.check_collision(mid_x, mid_y):
                        low = mid # Safe
                    else:
                        high = mid # Wall
                
                hit_dist = low
                hit_x = start_x + hit_dist * cos_a
                hit_y = start_y + hit_dist * sin_a
                return hit_dist, (int(hit_x), int(hit_y))
            
            distance = next_dist
            x, y = next_x, next_y
            
        # No hit
        end_x = start_x + max_distance * cos_a
        end_y = start_y + max_distance * sin_a
        return max_distance, (int(end_x), int(end_y))

    def compute_grid_positions(self, num_cars: int) -> List[GridSlot]:
        """
        Compute F1-style starting grid positions.
        
        Cars are placed on the center line going backward from the start,
        with small left/right stagger.
        
        Args:
            num_cars: Number of grid slots to generate
            
        Returns:
            List of GridSlot (x, y, angle) — all centered on the racing line
        """
        if not self.center_line:
            return []
        
        n = len(self.center_line)
        
        # Spacing between cars: ~3 center line points apart
        car_spacing = max(2, n // 150)
        
        positions = []
        for i in range(num_cars):
            # Go backward from start (point 0)
            idx = (n - (i + 1) * car_spacing) % n
            
            p = self.center_line[idx]
            heading = self.headings[idx]
            
            positions.append(GridSlot(x=p.x, y=p.y, angle=heading))
        
        self.grid_positions = positions
        return positions
    
    def compute_checkpoints(self, num_checkpoints: int = 4) -> list:
        """
        Auto-generate checkpoint lines evenly spaced around the track.
        
        Returns list of (start, end, index, is_finish_line) tuples.
        """
        if not self.center_line:
            return []
        
        n = len(self.center_line)
        checkpoints = []
        
        for i in range(num_checkpoints):
            # Evenly space around the track
            idx = int((i / num_checkpoints) * n) % n
            
            p = self.center_line[idx]
            heading = self.headings[idx]
            
            # Perpendicular to heading
            perp_x = -math.sin(heading)
            perp_y = math.cos(heading)
            
            # Checkpoint line spanning the track width
            w_right = self.raw_points[idx].w_right * self.scale
            w_left = self.raw_points[idx].w_left * self.scale
            
            start = (p.x - perp_x * w_left, p.y - perp_y * w_left)
            end = (p.x + perp_x * w_right, p.y + perp_y * w_right)
            
            is_finish = (i == 0)
            checkpoints.append((start, end, i, is_finish))
        
        self.checkpoint_lines = checkpoints
        return checkpoints
    
    def get_waypoints(self) -> list:
        """
        Get waypoints for the TrackProgressTracker.
        
        Returns list of dicts with keys: x, y, direction, track_progress
        """
        if not self.center_line:
            return []
        
        n = len(self.center_line)
        waypoints = []
        
        for i in range(n):
            waypoints.append({
                'x': self.center_line[i].x,
                'y': self.center_line[i].y,
                'direction': self.headings[i],
                'track_progress': i / n,
            })
        
        return waypoints
    
    def raycast(self, start_x: float, start_y: float, angle: float, 
                max_distance: float = 500.0) -> Tuple[float, Tuple[int, int]]:
        """
        Cast a ray against the track geometry (optimized).
        """
        # 1. Find approximate nearest segment index
        # Brute force search is okay-ish (1000 points), but we can optimize later
        # For now, let's just find the closest center point to start_x, start_y
        min_dist_sq = float('inf')
        nearest_idx = 0
        
        # Optimization: Only search if we moved significantly or first run
        # Ideally we cache this "current_segment" index in the agent/car
        # But here we don't have state.
        # We can implement a fast spatial lookup if this remains slow.
        
        # Fast search: Sample every 10th point first
        n = len(self.center_line)
        step = 10
        best_coarse_idx = 0
        
        for i in range(0, n, step):
            p = self.center_line[i]
            d2 = (p.x - start_x)**2 + (p.y - start_y)**2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_coarse_idx = i
                
        # Refine search locally
        min_dist_sq = float('inf')
        start_search = max(0, best_coarse_idx - step)
        end_search = min(n, best_coarse_idx + step)
        
        for i in range(start_search, end_search):
            p = self.center_line[i]
            d2 = (p.x - start_x)**2 + (p.y - start_y)**2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                nearest_idx = i

        # 2. Check collision against segments in a window logic
        # We only need to check segments that the ray MIGHT hit.
        # The ray goes in direction 'angle'.
        # We check segments starting from 'nearest_idx' and moving forward/backward?
        # A simple approach: Check nearest +/- 20 segments.
        # If max_distance is huge, we might hit something far away. 
        # But for racing, typically we hit the immediate walls.
        
        check_range = 30 # Check 30 segments ahead and behind
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # We step along the ray and check point_in_polygon for the candidate segments
        # Step size can be larger (e.g. 5px)
        step_size = 5.0
        dist = 0.0
        
        # Helper to check if point is in valid drivable area
        # We define drivable area as the union of quads.
        # If point is NOT in any quad, it is wall.
        # But we want to find the DISTANCE to the wall.
        # Ray starts presumably inside. We walk until we are OUTSIDE.
        
        # Optimized Walk:
        current_idx = nearest_idx
        
        while dist < max_distance:
            # Check point
            px = start_x + dist * cos_a
            py = start_y + dist * sin_a
            
            # Check if this point is inside any nearby quad
            in_track = False
            
            # We assume we move continuously, so we check near current_idx
            # We perform a local search for the segment this point belongs to
            # Search window shifts as we move
            
            # Search +/- 5 segments from current estimation
            found_segment = False
            for offset in range(-5, 6):
                idx = (current_idx + offset) % n
                next_i = (idx + 1) % n
                
                poly = [
                    (self.outer_boundary[idx].x, self.outer_boundary[idx].y),
                    (self.outer_boundary[next_i].x, self.outer_boundary[next_i].y),
                    (self.inner_boundary[next_i].x, self.inner_boundary[next_i].y),
                    (self.inner_boundary[idx].x, self.inner_boundary[idx].y)
                ]
                
                if self._point_in_polygon(px, py, poly):
                    in_track = True
                    found_segment = True
                    current_idx = idx # Update our tracker
                    break
            
            if not in_track:
                # We hit the wall!
                return dist, (int(px), int(py))
                
            dist += step_size
            
        return max_distance, (int(start_x + max_distance * cos_a), int(start_y + max_distance * sin_a))
    
    @staticmethod
    def list_available_tracks(tracks_dir: str = "tracks_data/tracks") -> List[str]:
        """List all available track names."""
        tracks = []
        if os.path.isdir(tracks_dir):
            for f in sorted(os.listdir(tracks_dir)):
                if f.endswith('.csv'):
                    tracks.append(os.path.splitext(f)[0])
        return tracks
