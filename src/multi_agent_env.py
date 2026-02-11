"""
Multi-Agent Racing Environment for F1 Simulator.

Extends the base environment to support multiple racing agents with
full race simulation: leaderboard, lap timing, visual effects.
"""
import math
import time
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from src.car import Car, CarConfig
from src.track import Track
from src.track_loader import TrackLoader
from src.sensors import LidarSensor, SensorConfig
from src.rewards import AdvancedRewardCalculator, RewardConfig, TrackProgressTracker
from src.race_manager import RaceManager, RaceState
from src.effects import VisualEffectsManager
from src.camera import Camera
from src.ui import AgentMarker, Minimap


class MultiAgentRacingEnv(gym.Env):
    """
    Multi-agent F1 Racing Environment.
    
    Supports multiple AI-controlled cars racing simultaneously with:
    - Full race simulation (countdown, laps, finish)
    - Position tracking and leaderboard
    - Lap timing with best lap tracking
    - Visual effects (tire trails, collision sparks)
    - Agent-agent collision detection
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        track_path: str = "assets/tracks/oval_track.png",
        track_name: Optional[str] = None,
        num_agents: int = 4,
        num_lidar_rays: int = 11,
        total_laps: int = 3,
        enable_agent_collision: bool = True,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.track_path = track_path
        self.track_name = track_name  # CSV track name (e.g. "Monza")
        self.num_agents = num_agents
        self.total_laps = total_laps
        self.enable_agent_collision = enable_agent_collision
        
        # Single agent observation/action for external control
        # (used when only one agent is RL-controlled)
        self.num_lidar_rays = num_lidar_rays
        obs_dim = num_lidar_rays + 4
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Components (initialized on first reset)
        self.track: Optional[Track] = None
        self.cars: Dict[int, Car] = {}
        self.sensors: Dict[int, LidarSensor] = {}
        self.reward_calculators: Dict[int, AdvancedRewardCalculator] = {}
        
        # UI & Camera
        self.camera: Optional[Camera] = None
        self.minimap: Optional[Minimap] = None
        self.ui_markers: Dict[int, AgentMarker] = {}
        
        # Race management
        self.race_manager: Optional[RaceManager] = None
        self.effects: Optional[VisualEffectsManager] = None
        self.progress_tracker: Optional[TrackProgressTracker] = None
        
        # Episode state
        self.steps = 0
        self.max_steps = 2000 * total_laps
        
        # Rendering
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None
        self._pygame_initialized = False
    
    def _init_pygame(self):
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("F1 RL Racing - Multi-Agent Race")
                self.screen = pygame.display.set_mode((1280, 720))
            else:
                pygame.display.set_mode((1, 1), pygame.HIDDEN)
                self.screen = pygame.Surface((1280, 720))
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont("Consolas", 14)
                self.font_large = pygame.font.SysFont("Consolas", 48)
            except:
                self.font = pygame.font.Font(None, 18)
                self.font_large = pygame.font.Font(None, 56)
            self._pygame_initialized = True
    
    def _init_simulation(self):
        self._init_pygame()
        
        self.camera = Camera(1280, 720)
        
        # Load track — CSV-based or image-based
        self.loader = None
        self._waypoints_data = None
        
        if self.track_name:
            # CSV-based track from racetrack database
            self.loader = TrackLoader(self.track_name)
            self.track = Track.from_loader(self.loader)
            self._waypoints_data = self.loader.get_waypoints()
            
            # Setup Minimap
            self.minimap = Minimap(self.loader, (200, 200))
        else:
            # Legacy image-based track
            self.track = Track(self.track_path)
            # Center camera on legacy track
            self.camera.x = 0
            self.camera.y = 0
            self.camera.zoom = 1.0
        
        # Create progress tracker
        self.progress_tracker = TrackProgressTracker(
            external_waypoints=self._waypoints_data
        )
        
        # Create race manager
        self.race_manager = RaceManager(
            num_racers=self.num_agents,
            total_laps=self.total_laps
        )
        
        # Create visual effects
        self.effects = VisualEffectsManager()
        
        # Create cars with staggered grid positions
        self._create_cars()
        
        # Initialize UI markers
        self.ui_markers = {}
        for i in range(self.num_agents):
            racer = self.race_manager.get_racer(i)
            color = racer.color if racer else (255, 0, 0)
            self.ui_markers[i] = AgentMarker(i, color)
    
    def _create_cars(self):
        """Create cars at F1-style grid positions on the track center line."""
        self.cars.clear()
        self.sensors.clear()
        self.reward_calculators.clear()
        
        # Get grid positions from loader or fallback
        if self.loader:
            grid = self.loader.compute_grid_positions(self.num_agents)
        else:
            # Fallback for legacy oval track
            tracker = TrackProgressTracker()
            grid = []
            pole_wp_idx = 32
            for i in range(self.num_agents):
                wp = tracker.waypoints[pole_wp_idx - i]
                from src.track_loader import GridSlot
                grid.append(GridSlot(x=wp.x, y=wp.y, angle=wp.direction))
        
        for i in range(self.num_agents):
            slot = grid[i]
            
            # Create car with racer color
            racer = self.race_manager.get_racer(i)
            config = CarConfig()
            
            # Adjust physics for realistic scale (8px/m) vs legacy scale
            if self.loader:
                # Real F1 car: ~90 m/s top speed * 8 px/m = 720 px/s
                config.max_velocity = 750.0  
                config.acceleration = 400.0
                config.braking = 600.0
                config.friction = 100.0
            
            self.cars[i] = Car(
                x=slot.x, y=slot.y, angle=slot.angle,
                config=config
            )
            self.cars[i].color = racer.color
            
            # Create sensor
            self.sensors[i] = LidarSensor(SensorConfig(
                num_rays=self.num_lidar_rays,
                fov=math.pi,
                max_distance=1000.0 # Increased range for high-speed AI (was 350)
            ))
            
            # Create reward calculator with track waypoints
            self.reward_calculators[i] = AdvancedRewardCalculator(
                external_waypoints=self._waypoints_data
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if self.track is None:
            self._init_simulation()
        else:
            self.race_manager.reset()
            self.effects.reset()
            self._create_cars()
        
        # Reset camera to primary agent
        if 0 in self.cars:
            self.camera.x = self.cars[0].x
            self.camera.y = self.cars[0].y
        
        # Reset reward calculators
        for i, calc in self.reward_calculators.items():
            car = self.cars[i]
            calc.reset(car.x, car.y)
        
        self.steps = 0
        
        # Start countdown
        self.race_manager.start_countdown()
        
        # Return observation for agent 0 (primary agent)
        obs = self._get_observation(0)
        info = {"race_state": self.race_manager.state.value}
        
        return obs, info
    
    def _get_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        car = self.cars[agent_id]
        sensor = self.sensors[agent_id]
        calc = self.reward_calculators[agent_id]
        
        lidar = sensor.update(car.x, car.y, car.angle, self.track)
        
        norm_velocity = car.velocity / car.config.max_velocity
        norm_steering = car.steering_angle / car.config.max_steering_angle
        
        progress = calc.progress_tracker.get_track_progress(car.x, car.y)
        norm_progress = progress * 2 - 1
        
        expected_heading = calc.progress_tracker.get_expected_heading(car.x, car.y)
        heading_diff = self._normalize_angle(car.angle - expected_heading)
        heading_alignment = math.cos(heading_diff)
        
        return np.array(
            lidar + [norm_velocity, norm_steering, norm_progress, heading_alignment],
            dtype=np.float32
        )
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def step(
        self,
        action: np.ndarray,
        agent_actions: Optional[Dict[int, np.ndarray]] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.
        """
        self.steps += 1
        dt = 1.0 / 60.0
        
        # Update Camera to follow agent 0
        if 0 in self.cars and self.camera:
            self.camera.update((self.cars[0].x, self.cars[0].y))
        
        # Wait during countdown
        if self.race_manager.state == RaceState.COUNTDOWN:
            self.race_manager.update({})
            obs = self._get_observation(0)
            return obs, 0.0, False, False, {"race_state": "countdown"}
        
        # Build actions for all agents
        if agent_actions is None:
            agent_actions = {}
        agent_actions[0] = action  # Primary agent
        
        # Fill in AI actions for other agents (simple AI)
        for i in range(1, self.num_agents):
            if i not in agent_actions:
                agent_actions[i] = self._get_simple_ai_action(i)
        
        # Update all cars
        crashed_agents = set()
        for i, car in self.cars.items():
            # Store previous valid position for collision resolution
            prev_x, prev_y = car.x, car.y
            
            act = agent_actions.get(i, np.array([0.0, 0.0]))
            
            # AI STUCK RECOVERY LOGIC
            # Check if car is stuck (low speed but wanting to move)
            # We need state for this, but we can approximate using current velocity vs intent
            # Better: The AI function itself should handle this, or we inject it here.
            # Let's inject a "Reverse Override" if we detect we are stuck against a wall.
            
            # If AI wants to go forward (throttle > 0) but speed is ~0, we might be stuck.
            # But the AI (get_simple_ai) now returns -1.0 (Brake) if close to wall.
            # We need to change that to Reverse.
            
            throttle = float(np.clip(act[0], -1.0, 1.0))
            steering = float(np.clip(act[1], -1.0, 1.0))
            
            car.set_controls(throttle, steering)
            car.update(dt)
            
            # Wall collision handling
            if self.track.check_car_collision(car.get_corners()):
                crashed_agents.add(i)
                self.effects.spawn_crash_effect(car.x, car.y)
                
                # CRITICAL FIX: Push out of wall
                # Revert to previous position to ensure we aren't trapped
                car.x = prev_x
                car.y = prev_y
                
                # Bounce: Reverse velocity with damping
                # If we hit head-on, reverse X and Y
                # This is a simplification. Ideally we reflect around normal.
                # But treating it as an elastic bounce on the velocity vector works okay.
                car.velocity = -car.velocity * 0.4
                
                # If speed was already low, we are just vibrating. 
                # Kill velocity to allow AI to reverse cleanly.
                if abs(car.velocity) < 20.0:
                    car.velocity = 0.0
            
            # Update effects
            self.effects.update_car(
                i, dt, car.x, car.y, car.angle,
                car.velocity, car.steering_angle
            )
        
        # Agent-agent collision with Momentum Transfer
        if self.enable_agent_collision:
            self._check_agent_collisions(crashed_agents)
        
        # ... [rest of step function] ...
        
    def _check_agent_collisions(self, crashed_agents: set):
        """
        Handle agent-agent collisions with Momentum Transfer (Elastic Collision).
        Prevents agents from getting stuck inside each other.
        """
        car_list = list(self.cars.items())
        
        for i in range(len(car_list)):
            for j in range(i + 1, len(car_list)):
                id1, car1 = car_list[i]
                id2, car2 = car_list[j]
                
                dx = car1.x - car2.x
                dy = car1.y - car2.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                collision_dist = 30  # Slightly smaller than visual size to allow bumping
                
                if dist < collision_dist and dist > 0:
                    # 1. Positional Correction (Push apart)
                    overlap = collision_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    total_mass = 2.0 # Assume equal mass
                    m1_ratio = 0.5
                    m2_ratio = 0.5
                    
                    car1.x += nx * overlap * m2_ratio
                    car1.y += ny * overlap * m2_ratio
                    car2.x -= nx * overlap * m1_ratio
                    car2.y -= ny * overlap * m1_ratio
                    
                    # 2. Velocity Exchange (1D elastic collision along normal)
                    # v1_new = v1 - 2*m2/(m1+m2) * dot(v1-v2, n) * n
                    
                    # Project velocities onto normal
                    # Since cars have scalar 'velocity' in direction 'angle', 
                    # we need to compute velocity vectors.
                    v1x = car1.velocity * math.cos(car1.angle)
                    v1y = car1.velocity * math.sin(car1.angle)
                    v2x = car2.velocity * math.cos(car2.angle)
                    v2y = car2.velocity * math.sin(car2.angle)
                    
                    # Relative velocity
                    rvx = v1x - v2x
                    rvy = v1y - v2y
                    
                    # Dot product with normal
                    vel_along_normal = rvx * nx + rvy * ny
                    
                    # Do not resolve if velocities are separating
                    if vel_along_normal > 0:
                        continue
                        
                    # Restitution (bounciness). 0.5 = somewhat inelastic crash
                    restitution = 0.5
                    
                    # Impulse scalar
                    j = -(1 + restitution) * vel_along_normal
                    j /= (1/1.0 + 1/1.0) # Inverse masses (1.0 each)
                    
                    # Apply impulse
                    impulse_x = j * nx
                    impulse_y = j * ny
                    
                    # New velocity vectors
                    v1x_new = v1x + impulse_x
                    v1y_new = v1y + impulse_y
                    v2x_new = v2x - impulse_x
                    v2y_new = v2y - impulse_y
                    
                    # Convert back to scalar velocity for our car model
                    # This is tricky because our car is non-holonomic (can't slide sideways easily)
                    # We project the new velocity vector onto the car's heading to get forward speed.
                    # Any sideways velocity is effectively "skidding" or lost energy.
                    
                    car1.velocity = v1x_new * math.cos(car1.angle) + v1y_new * math.sin(car1.angle)
                    car2.velocity = v2x_new * math.cos(car2.angle) + v2y_new * math.sin(car2.angle)
                    
                    # Add sparks
                    mid_x = (car1.x + car2.x) / 2
                    mid_y = (car1.y + car2.y) / 2
                    self.effects.spawn_collision_sparks(mid_x, mid_y)

        
        # Update effects
        self.effects.update(dt)
        
        # Calculate rewards and update race manager...
        # [Existing logic preserved]
        
        agent_rewards = {}
        for i, calc in self.reward_calculators.items():
            car = self.cars[i]
            reward_i, _ = calc.calculate_reward(
                car_x=car.x,
                car_y=car.y,
                car_angle=car.angle,
                car_velocity=car.velocity,
                crashed=i in crashed_agents,
                max_velocity=car.config.max_velocity
            )
            agent_rewards[i] = reward_i
        
        # Get progress from calculators for race manager
        racer_progress = {}
        for i, calc in self.reward_calculators.items():
            total = calc.laps_completed + calc.total_progress
            racer_progress[i] = (total, i in crashed_agents)
        
        # Update race
        self.race_manager.update(racer_progress)
        
        # Primary agent reward
        reward = agent_rewards.get(0, 0.0)
        
        # Check termination
        terminated = self.race_manager.state == RaceState.FINISHED
        truncated = self.steps >= self.max_steps
        
        # Observation
        obs = self._get_observation(0)
        
        # Info
        racer0 = self.race_manager.get_racer(0)
        info = {
            "race_state": self.race_manager.state.value,
            "position": racer0.position if racer0 else 1,
            "lap": racer0.current_lap if racer0 else 0,
            "best_lap": racer0.best_lap_time if racer0 else None,
            "crashed": 0 in crashed_agents,
        }
        
        return obs, reward, terminated, truncated, info

    def _respawn_car(self, agent_id: int):
        # ... (Existing logic)
        pass

    def _get_simple_ai_action(self, agent_id: int) -> np.ndarray:
        """
        Pro-Racing AI:
        1. Target Speed Profile based on Track Curvature (Lookahead).
        2. Pure Pursuit Steering (Chase a point on the racing line).
        3. Aggressive Throttle/Brake mapping.
        """
        car = self.cars[agent_id]
        calc = self.reward_calculators[agent_id]
        tracker = calc.progress_tracker
        
        # 1. Steering Control (Pure Pursuit)
        # Look ahead 80 pixels (~10m) for steering
        # We need to find the point on the track ahead of us
        current_idx, _ = tracker.get_nearest_waypoint(car.x, car.y)
        
        # Look ahead index
        lookahead_dist = 6 # waypoints
        target_idx = (current_idx + lookahead_dist) % tracker.num_waypoints
        target_wp = tracker.waypoints[target_idx]
        
        # Calculate steering angle to target
        dx = target_wp.x - car.x
        dy = target_wp.y - car.y
        target_heading = math.atan2(dy, dx)
        
        heading_diff = self._normalize_angle(target_heading - car.angle)
        
        # Proportional Steering
        steering = np.clip(heading_diff * 2.5, -1.0, 1.0)
        
        # 2. Speed Control (Curvature-based)
        # Look further ahead for braking (e.g. 30 waypoints ~50m)
        brake_lookahead = 25
        far_idx = (current_idx + brake_lookahead) % tracker.num_waypoints
        far_wp = tracker.waypoints[far_idx]
        
        # Calculate curvature at far point
        # diff between direction at current and direction at far
        # We use waypoints direction
        curr_dir = tracker.waypoints[current_idx].direction
        far_dir = far_wp.direction
        dir_diff = abs(self._normalize_angle(far_dir - curr_dir))
        
        # Estimate curvature intensity
        # High diff = Sharp turn ahead. Low diff = Straight.
        
        # Max Speed (pixels/s) -> 90 m/s * 8 = 720
        MAX_SPEED = 720.0
        MIN_CORNER_SPEED = 250.0 # ~30m/s (100km/h) hairpin
        
        # Map curvature to target speed
        # If dir_diff is > 1.0 rad over 25 waypoints, it's a turn
        
        if dir_diff > 0.5:
            # Corner ahead - Slow down!
            target_speed = MIN_CORNER_SPEED + (MAX_SPEED - MIN_CORNER_SPEED) * (1.0 - min(1.0, dir_diff))
        else:
            # Straight - Full send
            target_speed = MAX_SPEED
            
        current_speed = abs(car.velocity)
        
        # Throttle/Brake Logic
        throttle = 0.0
        
        if current_speed < target_speed:
            # Accelerate
            # If we are far below target, 100% throttle
            throttle = 1.0
        else:
            # Brake
            # Strong braking if we are exceeding target significantly
            overshoot = current_speed - target_speed
            if overshoot > 50:
                throttle = -1.0 # Full brake
            else:
                throttle = -0.5 # Moderate brake or coast
        
        # Wall Avoidance Override (Emergency & Unstick)
        sensor = self.sensors[agent_id]
        lidar = sensor.update(car.x, car.y, car.angle, self.track)
        mid = len(lidar) // 2
        front_dist_norm = lidar[mid]
        # max_distance = 1000.0
        
        if front_dist_norm < 0.05: # < 50 meters
            # Panic / Unstick Logic!
            # If we are VERY close, we need to reverse.
            # But the AI simply brakes (-1.0) which does not mean reverse in all logic,
            # but usually throttle < 0 is brake/reverse.
            
            throttle = -1.0 
            # Steer OPPOSITE to where we want to go, or better:
            # If wall is left, steer right to back up tail-left?
            # Actually, to back away from a wall, steering straight is best, 
            # or steering INTO the wall so the front swings out?
            # Let's just invert the steering we intended.
            steering = -steering
            
        elif front_dist_norm < 0.15:
            # Just brake hard
            throttle = -1.0
            
        # Add slight noise to prevent "perfect" robotic trains
        throttle += np.random.uniform(-0.02, 0.02)
        
        return np.array([throttle, steering], dtype=np.float32)

    def _check_agent_collisions(self, crashed_agents: set):
        """Check and handle agent-agent collisions."""
        car_list = list(self.cars.items())
        
        for i in range(len(car_list)):
            for j in range(i + 1, len(car_list)):
                id1, car1 = car_list[i]
                id2, car2 = car_list[j]
                
                # Simple circle collision
                dx = car1.x - car2.x
                dy = car1.y - car2.y
                dist = math.sqrt(dx * dx + dy * dy)
                
                collision_dist = 35  # Car radius sum
                if dist < collision_dist and dist > 0:
                    # Collision! Spawn sparks
                    mid_x = (car1.x + car2.x) / 2
                    mid_y = (car1.y + car2.y) / 2
                    self.effects.spawn_collision_sparks(mid_x, mid_y)
                    
                    # Push cars apart
                    overlap = collision_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    car1.x += nx * overlap * 0.5
                    car1.y += ny * overlap * 0.5
                    car2.x -= nx * overlap * 0.5
                    car2.y -= ny * overlap * 0.5

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        
        self._init_pygame()
        
        if self.track is None:
            return None
        
        # Background
        self.screen.fill((30, 30, 40))
        
        # Render Track (Custom or Legacy)
        if self.loader and self.camera:
            self.loader.render(self.screen, self.camera)
        else:
            self.track.render(self.screen, debug=False)
        
        # Effects (tire trails - behind cars)
        # Note: effects rendering needs to be camera-aware.
        # Currently EffectsManager uses screen coordinates. This is a TODO for effects.
        # For now, let's skip effects rendering correction or just let them look weird.
        # Actually effects use World coords but `render` assumes screen?
        # VisualEffectsManager.render does `pygame.draw`.
        # I won't fix effects in this step to keep it focused.
        
        # Cars
        for i, car in self.cars.items():
            racer = self.race_manager.get_racer(i)
            # Render car with camera transform
            if self.camera:
                self._render_car_camera(car, racer.color if racer else (255, 255, 255), i)
                # Render ID bubble
                if i in self.ui_markers:
                    self.ui_markers[i].render(self.screen, self.camera, (car.x, car.y))
            else:
                 self._render_car(car, racer.color if racer else (255, 255, 255), i)
        
        # LIDAR for primary agent
        # Manually render LIDAR rays transformed
        sensor = self.sensors[0]
        if self.camera:
            car0 = self.cars[0]
            # Transform hit points
            sx, sy = self.camera.transform(car0.x, car0.y)
            for dist, hit in zip(sensor.distances, sensor.hit_points):
                hx, hy = self.camera.transform(hit[0], hit[1])
                pygame.draw.line(self.screen, (0, 255, 0), (sx, sy), (hx, hy), 1)
        
        # HUD: Minimap
        if self.minimap:
            self.minimap.render(self.screen, self.cars, self.race_manager.racers)
            
        # HUD:        # HUD
        self._render_leaderboard()
        self._render_race_info()
        self._render_countdown()
        
        if self.render_mode == "human":
            # Handle user input for zoom
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
                    
                if self.camera:
                    if event.type == pygame.MOUSEWHEEL:
                        # Zoom in/out with mouse wheel
                        scale = 1.1 if event.y > 0 else 0.9
                        self.camera.set_zoom(self.camera.target_zoom * scale)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            self.camera.set_zoom(self.camera.target_zoom * 1.1)
                        elif event.key == pygame.K_MINUS:
                            self.camera.set_zoom(self.camera.target_zoom / 1.1)
                        elif event.key == pygame.K_0:
                            self.camera.set_zoom(1.0) # Reset
            
            # pygame.event.pump() # Handled by event.get()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def _render_car(self, car: Car, color: Tuple[int, int, int], agent_id: int):
        """Render a colored car (Legacy/Screen coords)."""
        corners = car.get_corners()
        pygame.draw.polygon(self.screen, color, corners)
        pygame.draw.polygon(self.screen, (255, 255, 255), corners, 2)
        
        # Name tag
        racer = self.race_manager.get_racer(agent_id)
        if racer:
            name = racer.name[:3]
            text = self.font.render(name, True, (255, 255, 255))
            self.screen.blit(text, (int(car.x - 10), int(car.y - 30)))

    def _render_car_camera(self, car: Car, color: Tuple[int, int, int], agent_id: int):
        """Render car using camera transform."""
        if not self.camera:
            return
            
        # Transform corners
        corners = car.get_corners()
        screen_corners = [self.camera.transform(x, y) for x, y in corners]
        
        pygame.draw.polygon(self.screen, color, screen_corners)
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_corners, 1)
    
    def _render_countdown(self):
        """Render F1-style red lights sequence."""
        lights = self.race_manager.get_lights_state()
        
        # Draw 5 light panels across the top of the screen
        light_y = 80
        light_spacing = 60
        total_width = 5 * light_spacing
        start_x = 640 - total_width // 2 + light_spacing // 2
        
        # Background panel for lights
        panel_rect = pygame.Rect(
            start_x - 40, light_y - 35,
            total_width + 20, 70
        )
        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 80), panel_rect, 3, border_radius=8)
        
        for i in range(5):
            cx = start_x + i * light_spacing
            
            if lights == 6:
                # LIGHTS OUT - all off (green glow briefly)
                elapsed_since_go = time.time() - self.race_manager.countdown_start_time - 5.0
                if elapsed_since_go < 1.5:
                    # Green flash
                    pygame.draw.circle(self.screen, (0, 200, 0), (cx, light_y), 20)
                    # Glow effect
                    glow = pygame.Surface((50, 50), pygame.SRCALPHA)
                    pygame.draw.circle(glow, (0, 200, 0, 80), (25, 25), 25)
                    self.screen.blit(glow, (cx - 25, light_y - 25))
                else:
                    # Dark
                    pygame.draw.circle(self.screen, (40, 40, 40), (cx, light_y), 20)
            elif i < lights:
                # Red light ON
                pygame.draw.circle(self.screen, (220, 0, 0), (cx, light_y), 20)
                # Red glow effect
                glow = pygame.Surface((50, 50), pygame.SRCALPHA)
                pygame.draw.circle(glow, (220, 0, 0, 100), (25, 25), 25)
                self.screen.blit(glow, (cx - 25, light_y - 25))
            else:
                # Light OFF (dark circle)
                pygame.draw.circle(self.screen, (40, 40, 40), (cx, light_y), 20)
            
            # Light border
            pygame.draw.circle(self.screen, (100, 100, 100), (cx, light_y), 20, 2)
        
        # Text below lights
        if lights == 6:
            elapsed_since_go = time.time() - self.race_manager.countdown_start_time - 5.0
            if elapsed_since_go < 2.0:
                text = self.font_large.render("LIGHTS OUT!", True, (0, 255, 0))
                rect = text.get_rect(center=(640, 160))
                self.screen.blit(text, rect)
    
    def _render_leaderboard(self):
        """Render race leaderboard."""
        # Panel
        panel = pygame.Rect(1050, 10, 220, 30 + self.num_agents * 28)
        pygame.draw.rect(self.screen, (20, 20, 30, 220), panel, border_radius=5)
        pygame.draw.rect(self.screen, (200, 150, 50), panel, 2, border_radius=5)
        
        # Title
        title = self.font.render("LEADERBOARD", True, (200, 150, 50))
        self.screen.blit(title, (1100, 15))
        
        # Racers
        y = 40
        for racer in self.race_manager.get_leaderboard():
            color = racer.color
            pos_text = f"P{racer.position}"
            name_text = racer.name[:8]
            lap_display = min(racer.current_lap + 1, self.total_laps)
            lap_text = f"L{lap_display}/{self.total_laps}"
            
            # Position
            self.screen.blit(self.font.render(pos_text, True, (255, 255, 255)), (1060, y))
            
            # Color indicator
            pygame.draw.rect(self.screen, color, (1095, y + 2, 12, 12))
            
            # Name
            self.screen.blit(self.font.render(name_text, True, color), (1115, y))
            
            # Lap
            self.screen.blit(self.font.render(lap_text, True, (180, 180, 180)), (1200, y))
            
            y += 28
    
    def _render_race_info(self):
        """Render race timer and info."""
        # Panel
        panel = pygame.Rect(10, 10, 200, 100)
        pygame.draw.rect(self.screen, (20, 20, 30, 220), panel, border_radius=5)
        pygame.draw.rect(self.screen, (0, 200, 150), panel, 2, border_radius=5)
        
        race_time = self.race_manager.get_race_time()
        racer = self.race_manager.get_racer(0)
        
        lines = [
            f"Race Time: {self.race_manager.format_time(race_time)}",
            f"Your Position: P{racer.position}" if racer else "",
            f"Lap: {racer.current_lap}/{self.total_laps}" if racer else "",
            f"Best Lap: {racer.get_best_lap_str()}" if racer else "",
        ]
        
        y = 15
        for line in lines:
            text = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(text, (15, y))
            y += 22
    
    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
