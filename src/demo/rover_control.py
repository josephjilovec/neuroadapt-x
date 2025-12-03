import pygame
import numpy as np
import math
import time
from typing import Optional

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAPTION = "NeuroAdapt-X: BCI Rover Control"
FPS = 60
BACKGROUND_COLOR = (20, 30, 40) # Dark space blue

ROVER_COLOR = (255, 100, 50)  # Orange/Red
ROVER_SIZE = 20
ROVER_SPEED = 2.0
ROTATION_SPEED = 3.0 # Degrees per frame

# --- Global State for BCI Input Simulation ---
# In a real application, this would be updated via a shared queue 
# or LSL outlet by the RealTimeProcessor.
LATEST_BCI_COMMAND: Optional[str] = None
COMMAND_EXPIRY_TIME = 0.5 # Seconds before command resets to 'Neutral'

class Rover:
    """
    Represents the simulated space rover, handling its position, 
    orientation, and drawing.
    """
    def __init__(self, x, y, angle):
        # Position (center of the rover)
        self.x = x
        self.y = y
        # Angle in degrees (0 = straight up)
        self.angle = angle 
        self.size = ROVER_SIZE
        self.last_command_time = time.time()

    def update(self, command: str):
        """
        Updates rover state based on the decoded BCI command.
        
        Args:
            command (str): 'LEFT', 'RIGHT', 'FORWARD', or 'NEUTRAL'.
        """
        # Command expiry logic
        if command != 'NEUTRAL':
            self.last_command_time = time.time()
        elif time.time() - self.last_command_time > COMMAND_EXPIRY_TIME:
            command = 'NEUTRAL'

        if command == 'LEFT':
            self.angle += ROTATION_SPEED
        elif command == 'RIGHT':
            self.angle -= ROTATION_SPEED
        
        # Always move forward slightly if commanded, based on orientation
        if command != 'NEUTRAL':
             # Convert angle to radians for trigonometric functions
            angle_rad = math.radians(self.angle + 90) # Add 90 for Pygame coordinate system (0 deg is usually right)
            
            # Use small movement regardless of rotation
            self.x += math.cos(angle_rad) * ROVER_SPEED * 0.2 
            self.y -= math.sin(angle_rad) * ROVER_SPEED * 0.2
            
        # Keep angle within 0-360 degrees
        self.angle %= 360
        
        # Keep rover within screen bounds
        self.x = np.clip(self.x, self.size, SCREEN_WIDTH - self.size)
        self.y = np.clip(self.y, self.size, SCREEN_HEIGHT - self.size)


    def draw(self, screen):
        """Draws the rover as a rotated triangle."""
        # Define the base shape of the rover (a triangle pointing up)
        # Coordinates are relative to the rover's center (0, 0)
        points = [
            (0, -self.size), # Nose
            (-self.size * 0.7, self.size * 0.5), # Left rear
            (self.size * 0.7, self.size * 0.5) # Right rear
        ]
        
        # Rotation Matrix application
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = []
        for px, py in points:
            # Rotate point
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # Translate to screen position
            rotated_points.append((int(self.x + rx), int(self.y + ry)))

        # Draw the main rover body
        pygame.draw.polygon(screen, ROVER_COLOR, rotated_points)
        
        # Draw a bright sensor at the front
        sensor_x, sensor_y = rotated_points[0]
        pygame.draw.circle(screen, (255, 255, 0), (sensor_x, sensor_y), 3)


def receive_bci_command(command: str):
    """
    Public function to be called by the RealTimeProcessor (via IPC)
    to update the rover's current instruction.
    """
    global LATEST_BCI_COMMAND
    LATEST_BCI_COMMAND = command
    
    # In a real implementation, this function would likely be part of a 
    # simple IPC server/client structure (e.g., using Python's multiprocessing.Queue 
    # or a socket connection) to communicate across the BCI thread and the Pygame thread.


def run_demo():
    """
    Initializes and runs the Pygame demo loop.
    """
    pygame.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(CAPTION)
    clock = pygame.time.Clock()
    
    # Initialize the Rover in the center
    rover = Rover(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 0)
    
    font = pygame.font.Font(None, 36)
    
    running = True
    current_command = 'NEUTRAL'
    
    print("\n--- Starting Rover Control Demo ---")
    print("Use ARROW KEYS for manual control or connect BCI processor.")
    
    while running:
        # --- 1. Input Handling ---
        manual_command = 'NEUTRAL'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Manual keyboard control for testing
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
        # Check current key states for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            manual_command = 'LEFT'
        elif keys[pygame.K_RIGHT]:
            manual_command = 'RIGHT'
        elif keys[pygame.K_UP]:
            manual_command = 'FORWARD' # Not used in this version but useful for expansion
        
        
        # --- 2. Determine Final Command ---
        # Prioritize BCI command if available and recently updated, otherwise use keyboard or neutral.
        global LATEST_BCI_COMMAND
        if LATEST_BCI_COMMAND:
            current_command = LATEST_BCI_COMMAND
            # Immediately reset the global variable after reading it
            LATEST_BCI_COMMAND = None 
        elif manual_command != 'NEUTRAL':
            current_command = manual_command
        else:
            current_command = 'NEUTRAL'

        # --- 3. Update State ---
        rover.update(current_command)
        
        # --- 4. Drawing ---
        screen.fill(BACKGROUND_COLOR)
        
        # Draw rover and environment
        rover.draw(screen)
        
        # Display status text
        text_surface = font.render(
            f"Command: {current_command} (Angle: {rover.angle:.1f}°)", 
            True, (255, 255, 255)
        )
        screen.blit(text_surface, (10, 10))
        
        text_info = font.render(
            "Use Arrow Keys or BCI | Press Q to Quit", 
            True, (150, 150, 150)
        )
        screen.blit(text_info, (10, 560))

        # 5. Update display and maintain frame rate
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("Rover Control Demo closed.")


if __name__ == '__main__':
    # When run directly, we simulate the environment where the BCI command 
    # would normally be passed.
    
    # NOTE: To run this demo successfully, you must have Pygame installed 
    # (pip install pygame).
    
    # For a simple mock BCI sequence:
    # 1. Start demo (it will be neutral)
    # 2. Wait 5 seconds
    # 3. Inject a 'LEFT' command
    
    import threading
    
    def mock_bci_injector():
        """Simulates an external BCI decoder sending commands."""
        time.sleep(5)
        print("MOCK BCI: Injecting 'LEFT' command...")
        receive_bci_command('LEFT')
        time.sleep(1.5)
        print("MOCK BCI: Injecting 'RIGHT' command...")
        receive_bci_command('RIGHT')
        time.sleep(2)
        print("MOCK BCI: Injecting 'LEFT' command...")
        receive_bci_command('LEFT')

    # Start the mock command injection thread
    injector_thread = threading.Thread(target=mock_bci_injector, daemon=True)
    # injector_thread.start() # Uncomment to see mock BCI input

    run_demo()
