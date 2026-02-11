
import pygame
import math
from typing import List, Tuple, Dict
from src.camera import Camera

class AgentMarker:
    """
    Renders a UI marker (bubble) above a car to identify agents.
    """
    def __init__(self, agent_id: int, color: Tuple[int, int, int]):
        self.agent_id = agent_id
        self.color = color
        self.radius = 12
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.text_surf = self.font.render(str(agent_id + 1), True, (255, 255, 255))
        self.text_rect = self.text_surf.get_rect()
        
    def render(self, surface: pygame.Surface, camera: Camera, world_pos: Tuple[float, float]):
        """
        Draw the marker at the given world position, transformed by camera.
        """
        wx, wy = world_pos
        sx, sy = camera.transform(wx, wy)
        
        # Only draw if roughly on screen (allow some margin for partial visibility)
        if -50 <= sx <= camera.width + 50 and -50 <= sy <= camera.height + 50:
            # Draw bubble above car
            # Offset upwards by ~40px (scaled?) -> No, constant screen offset
            # But we want it to float above the car regardless of zoom
            # So calculating screen pos of car, then subtracting constant Y
            
            # Car size on screen approx 40 * zoom
            # Let's put bubble slightly above that
            offset_y = 30
            
            # Draw circle background
            pygame.draw.circle(surface, self.color, (sx, sy - offset_y), self.radius)
            pygame.draw.circle(surface, (255, 255, 255), (sx, sy - offset_y), self.radius, 2)
            
            # Draw text
            self.text_rect.center = (sx, sy - offset_y)
            surface.blit(self.text_surf, self.text_rect)

class Minimap:
    """
    HUD Minimap that displays the full track and agent positions.
    """
    def __init__(self, track_loader, size: Tuple[int, int] = (200, 200)):
        self.width, self.height = size
        self.track_loader = track_loader
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        self.padding = 10
        
        # Pre-render track layout
        self._prerender_track()
        
    def _prerender_track(self):
        """Render the static track geometry to the minimap surface."""
        # Find track bounds to fit in minimap
        if not self.track_loader.center_line:
            return
            
        points = [(p.x, p.y) for p in self.track_loader.center_line]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        track_w = max_x - min_x
        track_h = max_y - min_y
        
        # Scale to fit minimap
        scale_x = (self.width - 2 * self.padding) / track_w if track_w > 0 else 1
        scale_y = (self.height - 2 * self.padding) / track_h if track_h > 0 else 1
        self.scale = min(scale_x, scale_y)
        
        # Calculate offsets to center
        scaled_w = track_w * self.scale
        scaled_h = track_h * self.scale
        
        self.offset_x = (self.width - scaled_w) / 2 - min_x * self.scale
        self.offset_y = (self.height - scaled_h) / 2 - min_y * self.scale
        
        # Draw transparent background
        self.surface.fill((0, 0, 0, 128))
        pygame.draw.rect(self.surface, (255, 255, 255), (0, 0, self.width, self.height), 2)
        
        # Draw track line
        scaled_points = []
        for x, y in points:
            sx = x * self.scale + self.offset_x
            sy = y * self.scale + self.offset_y
            scaled_points.append((sx, sy))
            
        if len(scaled_points) > 1:
            pygame.draw.lines(self.surface, (150, 150, 150), True, scaled_points, 3)
            
    def render(self, surface: pygame.Surface, cars: Dict[int, 'Car'], racers: Dict[int, 'RacerConfig']):
        """
        Draw the minimap and updated car positions.
        """
        # Blit the static track bg
        # Position: Bottom-left (20, 500) assuming 720p height
        pos = (20, 500)
        surface.blit(self.surface, pos)
        
        # Draw cars
        for agent_id, car in cars.items():
            # Convert car world pos to minimap pos
            mx = car.x * self.scale + self.offset_x + pos[0]
            my = car.y * self.scale + self.offset_y + pos[1]
            
            color = racers[agent_id].color if agent_id in racers else (255, 0, 0)
            
            # Draw dot
            pygame.draw.circle(surface, color, (int(mx), int(my)), 4)
            # Make player slightly larger
            if agent_id == 0:
                 pygame.draw.circle(surface, (255, 255, 255), (int(mx), int(my)), 5, 1)

