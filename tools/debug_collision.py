"""Debug script to check collision mask at spawn position."""
import pygame
import os

pygame.init()
pygame.display.set_mode((100, 100))  # Need a display for image loading

# Load mask
mask_path = 'assets/tracks/oval_track_mask.png'
img = pygame.image.load(mask_path).convert()
mask = pygame.mask.from_threshold(img, (255, 255, 255), (128, 128, 128))

# Check spawn position
cx, cy = 640, 360
spawn_y = cy + 200  # New spawn position (225 - 25 = 200)

print(f"Image size: {img.get_width()}x{img.get_height()}")
print(f"Spawn position: ({cx}, {spawn_y})")
print(f"Mask at spawn center: {mask.get_at((cx, int(spawn_y)))}")

# Check pixel color at spawn
print(f"Pixel color at spawn: {img.get_at((cx, int(spawn_y)))}")

# Check car corners (car is 40x20, facing up means front is -Y direction)
# When facing up (-90 degrees), car corners are:
# Front-right: (x + half_width, y - half_length) = (x+10, y-20)
# Front-left: (x - half_width, y - half_length) = (x-10, y-20)
# Rear-left: (x - half_width, y + half_length) = (x-10, y+20)
# Rear-right: (x + half_width, y + half_length) = (x+10, y+20)

corners = [
    (cx + 10, spawn_y - 20),  # Front right
    (cx - 10, spawn_y - 20),  # Front left
    (cx - 10, spawn_y + 20),  # Rear left
    (cx + 10, spawn_y + 20),  # Rear right
]

print("\nCar corners collision check:")
for i, (x, y) in enumerate(corners):
    corner_names = ["Front-right", "Front-left", "Rear-left", "Rear-right"]
    pixel = img.get_at((int(x), int(y)))
    mask_val = mask.get_at((int(x), int(y)))
    print(f"  {corner_names[i]} ({int(x)}, {int(y)}): pixel={pixel[:3]}, mask={mask_val}")

# Check a wider area around spawn
print("\nScanning Y values around spawn:")
for dy in range(-30, 31, 10):
    y = spawn_y + dy
    pixel = img.get_at((cx, int(y)))
    mask_val = mask.get_at((cx, int(y)))
    status = "WALL" if mask_val == 1 else "OK"
    print(f"  Y={int(y)}: pixel={pixel[:3]}, mask={mask_val} [{status}]")

pygame.quit()
