"""
Main entry point for the F1 Racing Simulator.
Phase 1: Physics Playground - Keyboard-controlled car with bicycle model physics.
"""
import pygame
import sys
from src.car import Car, CarConfig


# Window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# Colors
BG_COLOR = (30, 30, 40)
GRID_COLOR = (50, 50, 60)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 200, 150)


def draw_grid(surface: pygame.Surface, spacing: int = 50):
    """Draw a subtle grid for visual reference."""
    for x in range(0, WINDOW_WIDTH, spacing):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), 1)
    for y in range(0, WINDOW_HEIGHT, spacing):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), 1)


def draw_hud(surface: pygame.Surface, car: Car, font: pygame.font.Font):
    """Draw the heads-up display with car telemetry."""
    state = car.get_state()
    
    # Background panel
    panel_rect = pygame.Rect(10, 10, 250, 140)
    pygame.draw.rect(surface, (20, 20, 30, 200), panel_rect, border_radius=8)
    pygame.draw.rect(surface, ACCENT_COLOR, panel_rect, 2, border_radius=8)
    
    # Telemetry data
    lines = [
        f"Speed: {state['speed_kmh']:.1f} km/h",
        f"Velocity: {state['velocity']:.1f} px/s",
        f"Heading: {state['angle'] * 57.3:.1f}°",
        f"Steering: {state['steering_angle'] * 57.3:.1f}°",
        f"Position: ({state['x']:.0f}, {state['y']:.0f})",
    ]
    
    y_offset = 20
    for line in lines:
        text_surface = font.render(line, True, TEXT_COLOR)
        surface.blit(text_surface, (20, y_offset))
        y_offset += 24


def draw_controls_help(surface: pygame.Surface, font: pygame.font.Font):
    """Draw control hints at the bottom of the screen."""
    controls = "WASD or Arrow Keys to drive | R to reset | ESC to quit"
    text_surface = font.render(controls, True, (150, 150, 150))
    text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 25))
    surface.blit(text_surface, text_rect)


def handle_keyboard_input(car: Car, keys: pygame.key.ScancodeWrapper):
    """Convert keyboard input to car controls."""
    throttle = 0.0
    steering = 0.0
    
    # Throttle (W/Up = accelerate, S/Down = brake/reverse)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle = 1.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        throttle = -1.0
    
    # Steering (A/Left = left, D/Right = right)
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steering = -1.0
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steering = 1.0
    
    car.set_controls(throttle, steering)


def main():
    """Main game loop."""
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("F1 RL Racing Simulator - Phase 1: Physics Playground")
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    # Font for HUD
    try:
        font = pygame.font.SysFont("Consolas", 18)
    except:
        font = pygame.font.Font(None, 24)
    
    # Create car at center of screen
    car = Car(
        x=WINDOW_WIDTH // 2,
        y=WINDOW_HEIGHT // 2,
        angle=0,
        config=CarConfig()
    )
    
    # Main loop
    running = True
    debug_mode = True  # Show velocity/steering vectors
    
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset car to center
                    car.reset(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, 0)
                elif event.key == pygame.K_F1:
                    # Toggle debug mode
                    debug_mode = not debug_mode
        
        # Get keyboard state and apply to car
        keys = pygame.key.get_pressed()
        handle_keyboard_input(car, keys)
        
        # Update physics
        car.update(dt)
        
        # Keep car on screen (wrap around)
        if car.x < -50:
            car.x = WINDOW_WIDTH + 50
        elif car.x > WINDOW_WIDTH + 50:
            car.x = -50
        if car.y < -50:
            car.y = WINDOW_HEIGHT + 50
        elif car.y > WINDOW_HEIGHT + 50:
            car.y = -50
        
        # Render
        screen.fill(BG_COLOR)
        draw_grid(screen)
        car.render(screen, debug=debug_mode)
        draw_hud(screen, car, font)
        draw_controls_help(screen, font)
        
        # Title
        title = font.render("🏎️ Kinematic Bicycle Model Demo", True, ACCENT_COLOR)
        screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 15))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
