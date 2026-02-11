"""
Main entry point for the F1 Racing Simulator.
Phase 2: Environment & Sensors - Track with collisions and LIDAR.
"""
import pygame
import sys
import math
from src.car import Car, CarConfig
from src.track import Track
from src.sensors import LidarSensor, SensorConfig


# Window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Colors
BG_COLOR = (30, 30, 40)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 200, 150)
CRASH_COLOR = (255, 50, 50)


def draw_hud(surface: pygame.Surface, car: Car, font: pygame.font.Font, 
             lap_info: dict, crashed: bool):
    """Draw the heads-up display with car telemetry."""
    state = car.get_state()
    
    # Background panel
    panel_rect = pygame.Rect(10, 10, 260, 170)
    panel_color = (60, 20, 20, 220) if crashed else (20, 20, 30, 200)
    pygame.draw.rect(surface, panel_color, panel_rect, border_radius=8)
    border_color = CRASH_COLOR if crashed else ACCENT_COLOR
    pygame.draw.rect(surface, border_color, panel_rect, 2, border_radius=8)
    
    # Status
    status = "⚠️ CRASHED!" if crashed else "🏎️ Racing"
    status_surface = font.render(status, True, CRASH_COLOR if crashed else ACCENT_COLOR)
    surface.blit(status_surface, (20, 15))
    
    # Telemetry data
    lines = [
        f"Speed: {state['speed_kmh']:.1f} km/h",
        f"Heading: {math.degrees(state['angle']):.1f}°",
        f"Steering: {math.degrees(state['steering_angle']):.1f}°",
        f"Lap: {lap_info.get('lap', 0)}",
        f"Checkpoint: {lap_info.get('checkpoint', 0)}/4",
    ]
    
    y_offset = 45
    for line in lines:
        text_surface = font.render(line, True, TEXT_COLOR)
        surface.blit(text_surface, (20, y_offset))
        y_offset += 24


def draw_controls_help(surface: pygame.Surface, font: pygame.font.Font):
    """Draw control hints at the bottom of the screen."""
    controls = "WASD/Arrows: Drive | R: Reset | F1: Debug | ESC: Quit"
    text_surface = font.render(controls, True, (150, 150, 150))
    text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20))
    surface.blit(text_surface, text_rect)


def handle_keyboard_input(car: Car, keys: pygame.key.ScancodeWrapper):
    """Convert keyboard input to car controls."""
    throttle = 0.0
    steering = 0.0
    
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle = 1.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        throttle = -1.0
    
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steering = 1.0
    
    car.set_controls(throttle, steering)


def main():
    """Main game loop."""
    pygame.init()
    pygame.display.set_caption("F1 RL Racing Simulator - Phase 2: Environment & Sensors")
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.SysFont("Consolas", 18)
    except:
        font = pygame.font.Font(None, 24)
    
    # Load track
    track = Track("assets/tracks/oval_track.png")
    
    # Create car at spawn point
    spawn_x, spawn_y, spawn_angle = track.get_spawn_position()
    car = Car(x=spawn_x, y=spawn_y, angle=spawn_angle, config=CarConfig())
    
    # Create LIDAR sensor
    sensor = LidarSensor(SensorConfig(
        num_rays=11,
        fov=math.pi,  # 180 degree FOV
        max_distance=350.0
    ))
    
    # Game state
    crashed = False
    crash_timer = 0.0
    debug_mode = True
    
    # Lap tracking
    lap_info = {
        'lap': 0,
        'checkpoint': 0,
        'last_checkpoint_index': -1,
    }
    prev_pos = (car.x, car.y)
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset car
                    spawn_x, spawn_y, spawn_angle = track.get_spawn_position()
                    car.reset(spawn_x, spawn_y, spawn_angle)
                    crashed = False
                    crash_timer = 0.0
                    lap_info = {'lap': 0, 'checkpoint': 0, 'last_checkpoint_index': -1}
                elif event.key == pygame.K_F1:
                    debug_mode = not debug_mode
        
        # Handle crash state
        if crashed:
            crash_timer += dt
            if crash_timer > 1.5:  # Auto-reset after 1.5 seconds
                spawn_x, spawn_y, spawn_angle = track.get_spawn_position()
                car.reset(spawn_x, spawn_y, spawn_angle)
                crashed = False
                crash_timer = 0.0
                lap_info = {'lap': 0, 'checkpoint': 0, 'last_checkpoint_index': -1}
        else:
            # Get keyboard input and update car
            keys = pygame.key.get_pressed()
            handle_keyboard_input(car, keys)
            car.update(dt)
            
            # Check collision with track walls
            if track.check_car_collision(car.get_corners()):
                crashed = True
                car.velocity = 0
            
            # Check checkpoint crossings
            curr_pos = (car.x, car.y)
            for checkpoint in track.checkpoints:
                if track.check_checkpoint_crossing(prev_pos, curr_pos, checkpoint):
                    expected_index = (lap_info['last_checkpoint_index'] + 1) % len(track.checkpoints)
                    if checkpoint.index == expected_index:
                        lap_info['last_checkpoint_index'] = checkpoint.index
                        lap_info['checkpoint'] = checkpoint.index + 1
                        if checkpoint.is_finish_line and lap_info['checkpoint'] == len(track.checkpoints):
                            lap_info['lap'] += 1
                            lap_info['checkpoint'] = 0
            prev_pos = curr_pos
        
        # Update LIDAR sensor
        sensor.update(car.x, car.y, car.angle, track)
        
        # --- Rendering ---
        screen.fill(BG_COLOR)
        
        # Draw track
        track.render(screen, debug=debug_mode)
        
        # Draw LIDAR rays (before car so car renders on top)
        if debug_mode:
            sensor.render(screen, car.x, car.y)
        
        # Draw car
        car.render(screen, debug=debug_mode)
        
        # Draw crash flash
        if crashed:
            flash_alpha = int(100 * (1 - crash_timer / 1.5))
            if flash_alpha > 0:
                flash_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
                flash_surface.fill(CRASH_COLOR)
                flash_surface.set_alpha(flash_alpha)
                screen.blit(flash_surface, (0, 0))
        
        # Draw LIDAR minimap
        if debug_mode:
            sensor.render_minimap(screen, (WINDOW_WIDTH - 100, 100), scale=0.25)
        
        # Draw HUD
        draw_hud(screen, car, font, lap_info, crashed)
        draw_controls_help(screen, font)
        
        # Title
        title = font.render("🏎️ F1 RL Simulator - Phase 2: LIDAR & Collisions", True, ACCENT_COLOR)
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, WINDOW_HEIGHT - 50))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
