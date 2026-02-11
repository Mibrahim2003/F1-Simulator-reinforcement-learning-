"""
Visual Effects for F1 Racing Simulator.

Includes tire trails, collision sparks, and other visual polish effects.
"""
import pygame
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TireTrailPoint:
    """A single point in a tire trail."""
    x: float
    y: float
    age: float  # Seconds since created
    velocity: float  # Car velocity when created
    alpha: int = 255


@dataclass
class Particle:
    """A single particle effect."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    size: float
    life: float  # Remaining life in seconds
    max_life: float


class TireTrailEffect:
    """
    Creates tire trail marks behind the car.
    Shows skid marks when turning at high speed.
    """
    
    def __init__(
        self,
        max_points: int = 500,
        fade_time: float = 3.0,
        spawn_interval: float = 0.02
    ):
        self.max_points = max_points
        self.fade_time = fade_time
        self.spawn_interval = spawn_interval
        
        self.trails: List[TireTrailPoint] = []
        self.last_spawn_time: float = 0.0
    
    def update(
        self,
        dt: float,
        car_x: float,
        car_y: float,
        car_angle: float,
        car_velocity: float,
        steering_angle: float,
        car_width: float = 20.0
    ):
        """Update tire trails."""
        # Update existing trails
        self.trails = [
            TireTrailPoint(
                t.x, t.y, 
                t.age + dt, 
                t.velocity,
                max(0, int(255 * (1 - t.age / self.fade_time)))
            )
            for t in self.trails
            if t.age < self.fade_time
        ]
        
        # Spawn new trail points if moving fast and steering
        self.last_spawn_time += dt
        
        if self.last_spawn_time >= self.spawn_interval:
            self.last_spawn_time = 0.0
            
            # Only leave marks when sliding (high speed + steering)
            slide_factor = abs(car_velocity) * abs(steering_angle)
            if slide_factor > 50:  # Threshold for leaving marks
                # Calculate rear wheel positions
                rear_offset = 15  # Distance from center to rear
                wheel_offset = car_width / 2 - 3  # Distance from center to wheel
                
                rear_x = car_x - rear_offset * math.cos(car_angle)
                rear_y = car_y - rear_offset * math.sin(car_angle)
                
                # Left and right rear wheels
                perp_angle = car_angle + math.pi / 2
                
                left_x = rear_x + wheel_offset * math.cos(perp_angle)
                left_y = rear_y + wheel_offset * math.sin(perp_angle)
                
                right_x = rear_x - wheel_offset * math.cos(perp_angle)
                right_y = rear_y - wheel_offset * math.sin(perp_angle)
                
                self.trails.append(TireTrailPoint(left_x, left_y, 0, car_velocity))
                self.trails.append(TireTrailPoint(right_x, right_y, 0, car_velocity))
        
        # Limit trail count
        if len(self.trails) > self.max_points:
            self.trails = self.trails[-self.max_points:]
    
    def render(self, screen: pygame.Surface):
        """Render tire trails."""
        for trail in self.trails:
            if trail.alpha > 10:
                # Dark rubber color with fade
                color = (30, 30, 35)
                # Create a surface for alpha blending
                size = 4
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, trail.alpha), (size//2, size//2), size//2)
                screen.blit(s, (int(trail.x - size//2), int(trail.y - size//2)))


class ParticleSystem:
    """
    General purpose particle system for various effects.
    """
    
    def __init__(self, max_particles: int = 200):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
    
    def spawn_sparks(
        self,
        x: float,
        y: float,
        count: int = 15,
        velocity: float = 150.0
    ):
        """Spawn spark particles (for collisions)."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(velocity * 0.5, velocity)
            life = random.uniform(0.2, 0.5)
            
            # Orange/yellow spark colors
            color = random.choice([
                (255, 200, 50),
                (255, 150, 30),
                (255, 255, 100),
                (255, 100, 20)
            ])
            
            self.particles.append(Particle(
                x=x,
                y=y,
                vx=speed * math.cos(angle),
                vy=speed * math.sin(angle),
                color=color,
                size=random.uniform(2, 4),
                life=life,
                max_life=life
            ))
    
    def spawn_smoke(
        self,
        x: float,
        y: float,
        count: int = 5
    ):
        """Spawn smoke particles."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 50)
            life = random.uniform(0.5, 1.5)
            
            # Gray smoke
            gray = random.randint(80, 150)
            
            self.particles.append(Particle(
                x=x,
                y=y,
                vx=speed * math.cos(angle),
                vy=speed * math.sin(angle) - 30,  # Rise upward
                color=(gray, gray, gray),
                size=random.uniform(5, 10),
                life=life,
                max_life=life
            ))
    
    def update(self, dt: float):
        """Update all particles."""
        updated = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                # Gravity for sparks
                p.vy += 300 * dt
                # Shrink over time
                p.size *= 0.98
                updated.append(p)
        
        self.particles = updated[-self.max_particles:]
    
    def render(self, screen: pygame.Surface):
        """Render all particles."""
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            size = int(p.size)
            if size > 0 and alpha > 10:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p.color, alpha), (size, size), size)
                screen.blit(s, (int(p.x - size), int(p.y - size)))


class VisualEffectsManager:
    """
    Manages all visual effects for the racing simulation.
    """
    
    def __init__(self):
        self.tire_trails: dict = {}  # racer_id -> TireTrailEffect
        self.particles = ParticleSystem()
    
    def get_tire_trail(self, racer_id: int) -> TireTrailEffect:
        """Get or create tire trail for a racer."""
        if racer_id not in self.tire_trails:
            self.tire_trails[racer_id] = TireTrailEffect()
        return self.tire_trails[racer_id]
    
    def update_car(
        self,
        racer_id: int,
        dt: float,
        x: float,
        y: float,
        angle: float,
        velocity: float,
        steering: float
    ):
        """Update visual effects for a car."""
        trail = self.get_tire_trail(racer_id)
        trail.update(dt, x, y, angle, velocity, steering)
    
    def spawn_collision_sparks(self, x: float, y: float):
        """Spawn sparks at a collision point."""
        self.particles.spawn_sparks(x, y)
    
    def spawn_crash_effect(self, x: float, y: float):
        """Spawn full crash effect (sparks + smoke)."""
        self.particles.spawn_sparks(x, y, count=25, velocity=200)
        self.particles.spawn_smoke(x, y, count=10)
    
    def update(self, dt: float):
        """Update all effects."""
        self.particles.update(dt)
    
    def render(self, screen: pygame.Surface):
        """Render all effects."""
        # Render tire trails first (behind cars)
        for trail in self.tire_trails.values():
            trail.render(screen)
        
        # Render particles on top
        self.particles.render(screen)
    
    def reset(self):
        """Reset all effects."""
        self.tire_trails.clear()
        self.particles.particles.clear()
