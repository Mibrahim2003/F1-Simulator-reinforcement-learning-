"""
Script to generate a simple oval racing track for the F1 Simulator.
Uses Pillow to draw the track and save as PNG.
"""
from PIL import Image, ImageDraw
import os

def generate_track():
    # Image dimensions matching our window size
    WIDTH, HEIGHT = 1280, 720
    
    # Create a new image with transparency (RGBA)
    # Background is fully transparent (0 alpha)
    img = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Track colors
    TARMAC_COLOR = (40, 40, 45, 255)  # Dark gray
    GRASS_COLOR = (30, 100, 30, 255)   # Greenish (not used here, we keep transparency)
    WALL_COLOR = (200, 50, 50, 255)    # Red walls (for visual debug if needed)
    
    # Track geometry (Oval)
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    outer_radius_x, outer_radius_y = 500, 300
    track_width = 150  # Wider track for better collision tolerance
    
    # Draw the outer edge (Wall/Grass)
    # We actually want the 'drivable' area to be non-transparent?
    # Or 'walls' to be non-transparent?
    # Let's make the track mask: 
    #   Pixel = 1 (Solid) -> Wall/Collision
    #   Pixel = 0 (Transparent) -> Drivable
    
    # Actually, it's often easier to draw the VISUAL track first.
    # Let's draw the visual track: Gray road, Transparent background.
    
    # 1. Draw outer road boundary
    draw.ellipse(
        [
            center_x - outer_radius_x, center_y - outer_radius_y,
            center_x + outer_radius_x, center_y + outer_radius_y
        ],
        fill=TARMAC_COLOR,
        outline=None
    )
    
    # 2. Draw inner "hole" (Grass/Field) - modify alpha to making it transparent again?
    # Pillow doesn't easily support "erasing" with standard draw shapes unless using composite.
    # Easier strategy: Draw the road as a thick line.
    
    # Clear image
    img = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw simple oval using a thick stroke
    # Bounding box for the path
    bbox = [
        center_x - (outer_radius_x - track_width//2),
        center_y - (outer_radius_y - track_width//2),
        center_x + (outer_radius_x - track_width//2),
        center_y + (outer_radius_y - track_width//2)
    ]
    
    draw.ellipse(
        bbox,
        fill=None,
        outline=TARMAC_COLOR,
        width=track_width
    )
    
    # Also draw start/finish line
    start_x = center_x
    start_y = center_y + (outer_radius_y - track_width//2)
    draw.line(
        [start_x, start_y - track_width//2, start_x, start_y + track_width//2],
        fill=(255, 255, 255, 255),
        width=5
    )
    
    # Ensure assets directory exists
    os.makedirs('assets/tracks', exist_ok=True)
    
    # Save visual track
    img.save('assets/tracks/oval_track.png')
    print("Generated assets/tracks/oval_track.png")
    
    # --- Generate Collision Mask ---
    # For the mask, we want Walls = White (1), Drivable = Black (0)
    # This is inverse of the visual track which is Road = Color, Background = Transparent
    
    mask = Image.new('L', (WIDTH, HEIGHT), 255) # White background (Wall)
    draw_mask = ImageDraw.Draw(mask)
    
    # Draw the road as Black (Drivable)
    draw_mask.ellipse(
        bbox,
        fill=None,
        outline=0, # Black
        width=track_width
    )
    
    mask.save('assets/tracks/oval_track_mask.png')
    print("Generated assets/tracks/oval_track_mask.png")

if __name__ == "__main__":
    generate_track()
