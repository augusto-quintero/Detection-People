# ========================================================================
# PEOPLE TRACKING SYSTEM WITH YOLOV8 AND GPU
# WITH TIME TRACKING FOR ZONE CROSSINGS - SINGLE WINDOW VERSION
# ========================================================================
# This program detects and tracks people in a video using
# YOLO v8, with enhanced zone-based detection, crossing counting,
# and tracking of time spent between zones.
# Optimized for GPU processing on NVIDIA 1650Ti.
# ========================================================================

import cv2               # OpenCV for video and image processing
import os                # For file system operations
import time              # For performance timing
import torch             # Deep learning framework
import numpy as np       # For numerical operations
import subprocess        # For system commands
import sys               # For system information
import datetime          # For time formatting
from collections import defaultdict  # For tracking data storage
from ultralytics import YOLO  # YOLO object detection model

# ===== GPU DIAGNOSTICS =====
print("\n===== GPU DIAGNOSTICS =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Try to get CUDA information using subprocess
try:
    nvidia_smi = subprocess.check_output('nvidia-smi', shell=True)
    print("\nOutput from nvidia-smi:")
    print(nvidia_smi.decode('utf-8'))
    gpu_system_available = True
except:
    print("\nCouldn't run nvidia-smi. Checking other methods...")
    gpu_system_available = False

# ===== FORCE GPU USAGE =====
print("\n--- Configuring processing device ---")

# If CUDA is not available according to PyTorch but is at system level, try to force it
if not torch.cuda.is_available() and gpu_system_available:
    print("Attempting to force GPU detection...")
    try:
        # Force CUDA configurations
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force use of GPU 0
        torch.backends.cudnn.enabled = True
        # Try to initialize CUDA explicitly
        if hasattr(torch.cuda, 'init'):
            torch.cuda.init()
        print("After forcing: CUDA available =", torch.cuda.is_available())
    except Exception as e:
        print(f"Error forcing CUDA: {e}")

# Check if GPU is accessible
if torch.cuda.is_available():
    # Use GPU
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU available: {gpu_name}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory/1e9:.2f} GB")
    print(f"Using device: {device}")
    # Optimize CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    GPU_ENABLED = True
else:
    if gpu_system_available:
        print("\nWARNING! GPU detected at system level but PyTorch cannot access CUDA.")
        print("Possible solutions:")
        print("1. Reinstall PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("2. Verify you have updated NVIDIA drivers installed")
        print("3. If using a virtual environment, make sure PyTorch+CUDA is installed in this environment")
        print("\nAttempting to use GPU directly...")
        
        # Try to force GPU usage in Ultralytics model (last resort)
        device = 0  # Force index 0 for first GPU
        GPU_ENABLED = False  # Set as False for later logic
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")
        GPU_ENABLED = False

# Helper function for safe attribute access
def safe_get_attribute(obj, attr_name, default=None):
    """Extract an attribute safely, returning default value if it doesn't exist."""
    if hasattr(obj, attr_name):
        attr = getattr(obj, attr_name)
        return attr if attr is not None else default
    return default

# Create output directories for saving results
os.makedirs('detections', exist_ok=True)  # Directory for saving detection images
os.makedirs('videos', exist_ok=True)      # Directory for saving processed videos
os.makedirs('stats', exist_ok=True)       # Directory for saving statistical data

# ========== CONFIGURATION PARAMETERS ==========
VIDEO_PATH = "test.mp4"        # Path to the video file to process
MODEL = "yolov8l.pt"           # YOLOv8-large model (large version)
CONFIDENCE = 0.1               # Minimum confidence threshold for detections (10%)
SAVE_VIDEO = True              # If True, saves the video with detections
TRACK_PEOPLE_ONLY = True       # If True, only detects people (class 0)
IOU_THRESHOLD = 0.45           # IoU threshold to distinguish between nearby people
MAX_DETECTIONS = 100           # Maximum number of detections per frame
SHOW_LIVE_PREVIEW = True       # If True, shows live preview
MIN_HUMAN_HEIGHT = 10          # Minimum height (in pixels) to consider a person
MIN_HUMAN_WIDTH = 7           # Minimum width (in pixels) to consider a person
MAX_ASPECT_RATIO = 2.0         # Maximum aspect ratio (height/width)
MIN_ASPECT_RATIO = 0.5         # Minimum aspect ratio (height/width)
MOVEMENT_THRESHOLD = 0         # Movement threshold to distinguish static objects
# For 1650Ti (4GB RAM): Optimizations for limited memory
IMAGE_SIZE = 640               # Image size for processing (reduce if memory issues)
BATCH_SIZE = 1                 # Keep at 1 for GPUs with limited memory
MAIN_WINDOW_NAME = "Sistema de Seguimiento y Conteo"  # Nombre único para la ventana principal
# Zone tracking settings
ZONE_STABILITY_THRESHOLD = 2   # Frames required to confirm a zone change (reduces flicker)
CROSSING_MEMORY_TIME = 60      # Frames to remember a person's last known zone (for reappearing people)
DEBUG_MODE = True              # Enable debug output for zone crossings
# Show time statistics
SHOW_STATS_WINDOW = True       # Show detailed time statistics window
MAX_DISPLAYED_TIMES = 12       # Maximum number of crossing times to show in stats window
# ==================================================

print(f"Processing video: {VIDEO_PATH}")

# Initialize YOLO model for object detection with GPU
print(f"Loading model {MODEL}...")
try:
    # Specify the device when loading the model (force in case of detection problems)
    model = YOLO(MODEL)
    
    # Configure the model to use GPU if available
    if GPU_ENABLED:
        model.to(device)
    else:
        # Direct attempt even if PyTorch doesn't detect CUDA
        try:
            # Force device=0 in predict/train even if torch.cuda.is_available() is False
            print("Attempting to force GPU in model...")
            model.to('cuda:0')  # Direct attempt
        except Exception as e:
            print(f"Could not force GPU in model: {e}")
    
    print(f"Model loaded successfully")
except Exception as e:
    print(f"Could not load {MODEL}: {e}")
    print("Trying default model...")
    model = YOLO("yolov8n.pt")  # Lighter model as fallback
    
    # Try to assign to GPU even with fallback model
    if GPU_ENABLED:
        model.to(device)
    
    print(f"YOLOv8n model loaded as fallback")

# Open video for processing
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# Get basic video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Video width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
fps = cap.get(cv2.CAP_PROP_FPS)                         # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Total frames
print(f"Video loaded: {frame_width}x{frame_height} at {fps} FPS")
print(f"Total frames: {total_frames}")

# Video writer configuration for saving result
out = None  # Initialize variable to avoid errors
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video compression
    output_path = os.path.join("videos", "tracking_results.avi")
    
    try:
        # Create VideoWriter object with same dimensions and fps as original video
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise Exception("Could not open output file")
        print(f"Saving video as: {output_path}")
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        SAVE_VIDEO = False  # Disable video saving if error

# Variables for statistics and tracking
start_time = time.time()  # Start time for duration calculations
frame_count = 0  # Frame counter
track_history = defaultdict(list)  # Position history for each detected ID
total_persons_max = 0  # Maximum number of people detected in any frame
current_persons = 0  # Number of people in current frame
recent_persons = [0] * 10  # List for smoothing count (last 10 frames)
track_velocities = {}  # Dictionary to store movement speeds
static_object_ids = set()  # Set of IDs identified as static objects
verified_person_ids = set()  # Set of IDs confirmed as real people
processing_times = []  # List to store processing times per frame

# ====== ZONE DEFINITION VARIABLES ======
# Polygon selection variables
polygon_main = []  # Main detection area (orange)
polygon_zone_a = []  # Zone A (blue)
polygon_zone_b = []  # Zone B (blue)
current_polygon = 'main'  # Which polygon is being defined
polygon_selection_done = False  # Flag to indicate polygon selection is complete
in_zone_definition_mode = True  # Flag to track if we're in zone definition mode

# ===== IMPROVED ZONE TRACKING =====
# Dictionary to track which zone each person is in: 'a', 'b', 'transition', or None
person_in_zone = {}  
# Dictionary to store the last stable zone (only 'a' or 'b')
person_last_stable_zone = {}
# Dictionary to store each person's zone history for stability check
person_zone_history = defaultdict(list)
# Dictionary to store when a person was last seen in a particular zone
person_last_seen = {}
# Dictionary to store if a person has already been counted for a particular crossing
person_crossing_counted = defaultdict(set)  # Set of direction strings ("a_to_b", "b_to_a")
# Dictionary to store the frame number when a person was last seen
person_last_frame = {}
# Counts of zone transitions
zone_a_to_b_count = 0
zone_b_to_a_count = 0
# Special tracking for people who passed zone boundary
crossing_candidates = {}  # Track IDs that are candidates for crossing count
# Visualization data
visualization_data = []  # Store transition data for visualization
# Maximum time to keep visualization (in frames)
MAX_VISUALIZATION_TIME = 30
# Store detailed crossing data for debugging/analysis
crossings_log = []

# ===== NEW: TIME TRACKING FOR CROSSINGS =====
# Track when each person enters Zone A (frame number)
person_entered_zone_a = {}
# Track when each person enters Zone B (frame number)
person_entered_zone_b = {}
# Track crossing times from A to B (in seconds)
crossing_times_a_to_b = {}  # Dictionary mapping ID to time in seconds
# Track crossing times from B to A (in seconds)
crossing_times_b_to_a = {}  # Dictionary mapping ID to time in seconds
# Store all completed crossings with timestamps for statistics
all_crossings = []  # List of dicts with crossing details

# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point: Tuple (x, y) representing the point
        polygon: List of tuples [(x1, y1), (x2, y2), ...] representing polygon vertices
        
    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    if len(polygon) < 3:  # Polygon must have at least 3 points
        return False
        
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# Function to handle mouse events for polygon selection
def select_polygon(event, x, y, flags, param):
    global polygon_main, polygon_zone_a, polygon_zone_b, current_polygon, polygon_selection_done
    
    # Solo procesar eventos si estamos en modo de definición de zonas
    if not in_zone_definition_mode:
        return
        
    # If event is left button click, add point to current polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_polygon == 'main' and len(polygon_main) < 4:
            polygon_main.append((x, y))
            print(f"Main zone point {len(polygon_main)} added at ({x}, {y})")
            if len(polygon_main) == 4:
                current_polygon = 'zone_a'
                print("Now define Zone A (blue) - click 4 points")
        
        elif current_polygon == 'zone_a' and len(polygon_zone_a) < 4:
            polygon_zone_a.append((x, y))
            print(f"Zone A point {len(polygon_zone_a)} added at ({x}, {y})")
            if len(polygon_zone_a) == 4:
                current_polygon = 'zone_b'
                print("Now define Zone B (blue) - click 4 points")
        
        elif current_polygon == 'zone_b' and len(polygon_zone_b) < 4:
            polygon_zone_b.append((x, y))
            print(f"Zone B point {len(polygon_zone_b)} added at ({x}, {y})")
            if len(polygon_zone_b) == 4:
                current_polygon = 'done'
                polygon_selection_done = True
                print("All zones defined! Press any key to continue...")

# Function to determine if a person is in a zone using multiple points
def determine_zone(x1, y1, x2, y2, track_id=None):
    """
    Determine which zone a person is in using multiple points on their bounding box.
    Uses a more robust approach with center of mass and multiple points.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        track_id: Person's tracking ID
        
    Returns:
        str: 'a', 'b', 'transition', or None
    """
    # Create multiple points to check
    points = [
        ((x1 + x2) // 2, y2),                      # Bottom center (feet)
        ((x1 + x2) // 2, (y1 + y2) // 2),          # Center of mass (most important)
        ((x1 + x2) // 2, (y1 + y2) // 2 + (y2-y1)//4),  # Lower center (hip level)
        (x1 + (x2-x1)//4, y2),                     # Bottom quarter from left
        (x2 - (x2-x1)//4, y2)                      # Bottom quarter from right
    ]
    
    # Count how many points are in each zone
    zone_a_points = 0
    zone_b_points = 0
    
    for point in points:
        if point_in_polygon(point, polygon_zone_a):
            zone_a_points += 1
        if point_in_polygon(point, polygon_zone_b):
            zone_b_points += 1
    
    # Weight the center of mass point more heavily (double count it)
    center_in_a = point_in_polygon(((x1 + x2) // 2, (y1 + y2) // 2), polygon_zone_a)
    center_in_b = point_in_polygon(((x1 + x2) // 2, (y1 + y2) // 2), polygon_zone_b)
    
    if center_in_a:
        zone_a_points += 2
    if center_in_b:
        zone_b_points += 2
    
    # Decision logic - get the current frame's raw zone assignment
    if zone_a_points > 0 and zone_b_points > 0:
        # Person is in both zones, determine which one has more points
        if zone_a_points > zone_b_points:
            raw_zone = 'a'
        elif zone_b_points > zone_a_points:
            raw_zone = 'b'
        else:
            # Equal number of points in both zones
            raw_zone = 'transition'
    elif zone_a_points > 0:
        raw_zone = 'a'
    elif zone_b_points > 0:
        raw_zone = 'b'
    else:
        raw_zone = None  # Not in any zone
    
    # If we have a track_id, we can apply stability checks to avoid zone flickering
    if track_id is not None:
        # Update zone history (keep last 5 frames only)
        person_zone_history[track_id].append(raw_zone)
        if len(person_zone_history[track_id]) > 5:
            person_zone_history[track_id].pop(0)
        
        # Check if history is consistent
        history = person_zone_history[track_id]
        
        # If we have a consistent zone in history, use it
        if len(history) >= ZONE_STABILITY_THRESHOLD:
            recent_history = history[-ZONE_STABILITY_THRESHOLD:]
            if all(z == 'a' for z in recent_history):
                # Update last stable zone when we have a stable detection
                person_last_stable_zone[track_id] = 'a'
                person_last_seen[track_id] = frame_count
                
                # NEW: Track when the person first enters Zone A
                if track_id not in person_entered_zone_a:
                    person_entered_zone_a[track_id] = frame_count
                
                return 'a'
            elif all(z == 'b' for z in recent_history):
                # Update last stable zone when we have a stable detection
                person_last_stable_zone[track_id] = 'b'
                person_last_seen[track_id] = frame_count
                
                # NEW: Track when the person first enters Zone B
                if track_id not in person_entered_zone_b:
                    person_entered_zone_b[track_id] = frame_count
                
                return 'b'
        
        # If not enough history or no consistency, return current raw zone
        return raw_zone
    else:
        # If no track_id, just return raw zone
        return raw_zone

# Function to format time in seconds to MM:SS.ss format
def format_time(seconds):
    """Format time in seconds to MM:SS.ss format"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    return f"{minutes:02d}:{seconds_remainder:05.2f}"

# Function to determine if a person has crossed between zones and track the timing
def check_zone_crossing(track_id, current_zone, x1, y1, x2, y2):
    """
    Check if a person has crossed from one zone to another and calculate the time.
    
    Args:
        track_id: Person's tracking ID
        current_zone: Current zone ('a', 'b', 'transition', or None)
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        tuple: (crossed, direction) where crossed is bool and direction is string or None
    """
    global zone_a_to_b_count, zone_b_to_a_count, person_crossing_counted, crossing_times_a_to_b, crossing_times_b_to_a
    
    # Record the current frame as last seen for this ID
    person_last_frame[track_id] = frame_count
    
    # If current_zone is None or 'transition', no crossing check needed
    if current_zone is None or current_zone == 'transition':
        return (False, None)
    
    # Get the last stable zone for this track_id
    previous_zone = person_last_stable_zone.get(track_id)
    
    # If no previous stable zone, set current as first and return
    if previous_zone is None:
        person_last_stable_zone[track_id] = current_zone
        person_last_seen[track_id] = frame_count
        
        # Record first zone entry time
        if current_zone == 'a' and track_id not in person_entered_zone_a:
            person_entered_zone_a[track_id] = frame_count
        elif current_zone == 'b' and track_id not in person_entered_zone_b:
            person_entered_zone_b[track_id] = frame_count
            
        return (False, None)
        
    # Handle case where current zone is same as last stable zone
    if current_zone == previous_zone:
        # Update last seen time
        person_last_seen[track_id] = frame_count
        return (False, None)
    
    # We have a zone change! Determine direction
    if previous_zone == 'a' and current_zone == 'b':
        direction = "a_to_b"
    elif previous_zone == 'b' and current_zone == 'a':
        direction = "b_to_a"
    else:
        # Unknown transition
        return (False, None)
    
    # Check if this direction has already been counted for this ID
    if direction in person_crossing_counted[track_id]:
        # We already counted this ID crossing in this direction
        if DEBUG_MODE:
            print(f"ID {track_id} already counted crossing {direction}, ignoring")
        
        # Still update the stable zone and last seen time
        person_last_stable_zone[track_id] = current_zone
        person_last_seen[track_id] = frame_count
        return (False, None)
    
    # Valid new crossing detected!
    
    # NEW: Calculate crossing time for A to B
    if direction == "a_to_b" and track_id in person_entered_zone_a:
        entry_frame = person_entered_zone_a[track_id]
        exit_frame = frame_count
        
        # Calculate time in seconds using the frame rate
        time_seconds = (exit_frame - entry_frame) / fps
        
        # Store the crossing time
        crossing_times_a_to_b[track_id] = time_seconds
        
        # Add to all crossings for statistics
        all_crossings.append({
            'id': track_id,
            'direction': 'a_to_b',
            'start_frame': entry_frame,
            'end_frame': exit_frame,
            'time_seconds': time_seconds,
            'formatted_time': format_time(time_seconds)
        })
        
        if DEBUG_MODE:
            print(f"Person ID {track_id} took {format_time(time_seconds)} to cross from Zone A to Zone B")
    
    # NEW: Calculate crossing time for B to A
    if direction == "b_to_a" and track_id in person_entered_zone_b:
        entry_frame = person_entered_zone_b[track_id]
        exit_frame = frame_count
        
        # Calculate time in seconds using the frame rate
        time_seconds = (exit_frame - entry_frame) / fps
        
        # Store the crossing time
        crossing_times_b_to_a[track_id] = time_seconds
        
        # Add to all crossings for statistics
        all_crossings.append({
            'id': track_id,
            'direction': 'b_to_a',
            'start_frame': entry_frame,
            'end_frame': exit_frame,
            'time_seconds': time_seconds,
            'formatted_time': format_time(time_seconds)
        })
        
        if DEBUG_MODE:
            print(f"Person ID {track_id} took {format_time(time_seconds)} to cross from Zone B to Zone A")
    
    # Record details of the crossing for debugging
    crossing_details = {
        'id': track_id,
        'direction': direction,
        'frame': frame_count,
        'previous_zone': previous_zone,
        'current_zone': current_zone,
        'position': ((x1+x2)//2, (y1+y2)//2)
    }
    crossings_log.append(crossing_details)
    
    # Count the crossing based on direction
    if direction == "a_to_b":
        zone_a_to_b_count += 1
        if DEBUG_MODE:
            print(f"\n¡CRUCE A→B DETECTADO! Persona ID {track_id} cruzó de Zona A a Zona B (frame {frame_count})")
            print(f"Contador A→B = {zone_a_to_b_count}")
    elif direction == "b_to_a":
        zone_b_to_a_count += 1
        if DEBUG_MODE:
            print(f"\n¡CRUCE B→A DETECTADO! Persona ID {track_id} cruzó de Zona B a Zona A (frame {frame_count})")
            print(f"Contador B→A = {zone_b_to_a_count}")
    
    # Add this crossing to list of counted crossings for this ID
    person_crossing_counted[track_id].add(direction)
    
    # Update stable zone and last seen
    person_last_stable_zone[track_id] = current_zone
    person_last_seen[track_id] = frame_count
    
    # Create visualization data
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Get a suitable start point for visualization arrow
    if track_id in track_history and len(track_history[track_id]) >= 10:
        # Use a point from recent history
        start_point = track_history[track_id][-10]  # 10 frames ago
    else:
        # Use approximate position offset
        offset_x = -50 if direction == "b_to_a" else 50
        start_point = (center_x + offset_x, center_y)
    
    # Add visualization
    if direction == "a_to_b":
        visualization_data.append(('A_to_B', start_point, (center_x, center_y), 0))
    else:
        visualization_data.append(('B_to_A', start_point, (center_x, center_y), 0))
        
    return (True, direction)

# Function to draw polygons and tracking information on frame
def draw_polygons(frame):
    """Draw the defined polygons and tracking information on the frame"""
    # Draw main detection zone (orange)
    if len(polygon_main) > 0:
        for i in range(len(polygon_main)):
            # Draw vertices as circles
            cv2.circle(frame, polygon_main[i], 5, (0, 165, 255), -1)  # Orange
            # Draw edges
            if i < len(polygon_main) - 1:
                cv2.line(frame, polygon_main[i], polygon_main[i+1], (0, 165, 255), 2)
        # Draw closing line if we have at least 2 points
        if len(polygon_main) >= 2:
            cv2.line(frame, polygon_main[-1], polygon_main[0], (0, 165, 255), 2)
    
    # Draw zone A (blue)
    if len(polygon_zone_a) > 0:
        for i in range(len(polygon_zone_a)):
            # Draw vertices as circles
            cv2.circle(frame, polygon_zone_a[i], 5, (255, 0, 0), -1)  # Blue
            # Draw edges
            if i < len(polygon_zone_a) - 1:
                cv2.line(frame, polygon_zone_a[i], polygon_zone_a[i+1], (255, 0, 0), 2)
        # Draw closing line if we have at least 2 points
        if len(polygon_zone_a) >= 2:
            cv2.line(frame, polygon_zone_a[-1], polygon_zone_a[0], (255, 0, 0), 2)
    
    # Draw zone B (blue)
    if len(polygon_zone_b) > 0:
        for i in range(len(polygon_zone_b)):
            # Draw vertices as circles
            cv2.circle(frame, polygon_zone_b[i], 5, (255, 0, 0), -1)  # Blue
            # Draw edges
            if i < len(polygon_zone_b) - 1:
                cv2.line(frame, polygon_zone_b[i], polygon_zone_b[i+1], (255, 0, 0), 2)
        # Draw closing line if we have at least 2 points
        if len(polygon_zone_b) >= 2:
            cv2.line(frame, polygon_zone_b[-1], polygon_zone_b[0], (255, 0, 0), 2)
    
    # Add labels to zones
    if len(polygon_zone_a) == 4:
        # Calculate center of zone A
        center_x = sum(p[0] for p in polygon_zone_a) // 4
        center_y = sum(p[1] for p in polygon_zone_a) // 4
        cv2.putText(frame, "ZONE A", (center_x-40, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if len(polygon_zone_b) == 4:
        # Calculate center of zone B
        center_x = sum(p[0] for p in polygon_zone_b) // 4
        center_y = sum(p[1] for p in polygon_zone_b) // 4
        cv2.putText(frame, "ZONE B", (center_x-40, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Add instruction text during zone definition
    if in_zone_definition_mode:
        instructions = ""
        if current_polygon == 'main':
            instructions = "DEFINE PRINCIPAL ZONE (ORANGE) - CLICK ON 4 POINTS"
        elif current_polygon == 'zone_a':
            instructions = "DEFINE ZONE A (BLUE) - CLICK ON 4 POINTS"
        elif current_polygon == 'zone_b':
            instructions = "DEFINE ZONE B (BLUE) - CLICK ON 4 POINTS"
        elif current_polygon == 'done':
            instructions = "ALL ZONES DEFINED! PRESS ANY KEY TO CONTINUE..."
        
        cv2.putText(frame, instructions, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw transition visualization arrows
    for item in visualization_data:
        # Unpack visualization data
        transition_type, start_point, end_point, frames_ago = item
        
        # Only show transitions for a limited time
        if frames_ago < MAX_VISUALIZATION_TIME:
            if transition_type == 'A_to_B':
                # Draw a prominent green arrow for A to B transition
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 3)
            elif transition_type == 'B_to_A':
                # Draw a prominent yellow arrow for B to A transition
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 255), 3)
    
    # Add counter information with more prominent display
    if not in_zone_definition_mode:
        # Background for counter display
        cv2.rectangle(frame, (5, 45), (230, 145), (0, 0, 0), -1)
        
        # Draw counter values with larger font and prominent colors
        cv2.putText(frame, f"A -> B: {zone_a_to_b_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"B -> A: {zone_b_to_a_count}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    
        # Show average crossing time if available
        if len(crossing_times_a_to_b) > 0:
            avg_time = sum(crossing_times_a_to_b.values()) / len(crossing_times_a_to_b)
            cv2.putText(frame, f"T. Prom: {format_time(avg_time)}s", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Create detailed statistics window
def create_stats_window():
    """Create a window to display crossing time statistics"""
    # Create a black background image
    stats_height = 600
    stats_width = 400
    stats_image = np.zeros((stats_height, stats_width, 3), np.uint8)
    
    # Add title
    cv2.putText(stats_image, "CROSSING STATISTICS", (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add section headers
    cv2.putText(stats_image, "Crossings A -> B:", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    # Display crossing times from A to B
    y_pos = 100
    count = 0
    if crossing_times_a_to_b:
        # Sort by crossing time (fastest first)
        sorted_times = sorted(crossing_times_a_to_b.items(), key=lambda x: x[1])
        for person_id, time_seconds in sorted_times:
            if count < MAX_DISPLAYED_TIMES:  # Limit the number of entries
                time_str = format_time(time_seconds)
                text = f"Persona #{person_id}: {time_str}"
                cv2.putText(stats_image, text, (40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_pos += 25
                count += 1
        
        # Add average time
        avg_time = sum(crossing_times_a_to_b.values()) / len(crossing_times_a_to_b)
        cv2.putText(stats_image, f"Average time: {format_time(avg_time)}", (40, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add fastest time
        fastest_time = min(crossing_times_a_to_b.values()) if crossing_times_a_to_b else 0
        cv2.putText(stats_image, f"Fastest time: {format_time(fastest_time)}", (40, y_pos + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(stats_image, "No crossing data available", (40, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Add section headers for B to A
    cv2.putText(stats_image, "Crossings B -> A:", (20, y_pos + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
    # Display crossing times from B to A
    y_pos += 110
    count = 0
    if crossing_times_b_to_a:
        # Sort by crossing time (fastest first)
        sorted_times = sorted(crossing_times_b_to_a.items(), key=lambda x: x[1])
        for person_id, time_seconds in sorted_times:
            if count < MAX_DISPLAYED_TIMES:  # Limit the number of entries
                time_str = format_time(time_seconds)
                text = f"Persona #{person_id}: {time_str}"
                cv2.putText(stats_image, text, (40, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                y_pos += 25
                count += 1
        
        # Add average time
        avg_time = sum(crossing_times_b_to_a.values()) / len(crossing_times_b_to_a)
        cv2.putText(stats_image, f"Average time: {format_time(avg_time)}", (40, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add fastest time
        fastest_time = min(crossing_times_b_to_a.values()) if crossing_times_b_to_a else 0
        cv2.putText(stats_image, f"Fastest time: {format_time(fastest_time)}", (40, y_pos + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(stats_image, "No crossing data available", (40, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Add current time and frame number
    current_time = frame_count / fps if fps > 0 else 0
    cv2.putText(stats_image, f"Video time: {format_time(current_time)}", (20, stats_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(stats_image, f"Frame: {frame_count}/{total_frames}", (20, stats_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return stats_image

# Function to enhance image quality before detection
def enhance_frame(frame):
    """
    Enhances the frame for better people detection in various conditions.
    
    Args:
        frame: Original frame to process
        
    Returns:
        enhanced: Frame with improved contrast and sharpness
    """
    try:
        enhanced = frame.copy()  # Create copy to avoid modifying original
        
        # Apply adaptive contrast enhancement (CLAHE)
        # Convert to LAB color space (Luminosity, A, B)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  # Split channels
        # Apply CLAHE only to luminosity channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # Recombine channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening filter to enhance edges
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # Sharpness kernel
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return frame  # Return original image if error

# Function to validate if a detection corresponds to a real person
def is_valid_human(x1, y1, x2, y2, track_id=None):
    """
    Validates if a detection has geometric characteristics of a person.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        track_id: Object tracking ID
        
    Returns:
        bool: True if the detection appears to be a person, False otherwise
    """
    try:
        # Calculate dimensions
        width = x2 - x1   # Box width
        height = y2 - y1  # Box height
        
        # If this ID was previously verified as a person, accept it directly
        if track_id is not None and track_id in verified_person_ids:
            return True
        
        # Check minimum size to filter out objects that are too small
        if height < MIN_HUMAN_HEIGHT or width < MIN_HUMAN_WIDTH:
            return False
        
        # Check aspect ratio (height/width)
        # People are typically taller than wide
        if width <= 0:  # Prevent division by zero
            return False
            
        aspect_ratio = height / width
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return False
        
        # If passed all checks, mark this ID as a verified person
        if track_id is not None:
            verified_person_ids.add(track_id)
        
        return True
    except Exception as e:
        print(f"Error in human validation: {e}")
        return False  # When in doubt, don't consider as a person

# Function to calculate average movement of an object
def calculate_movement(track_history, track_id):
    """
    Calculates the average movement magnitude of an object between frames.
    
    Args:
        track_history: Position history of all objects
        track_id: ID of the object to analyze
        
    Returns:
        float: Average movement magnitude in pixels
    """
    try:
        # If not enough history, return default value
        if track_id not in track_history or len(track_history[track_id]) <= 3:
            return 10  # Default value if not enough points
        
        # Calculate average movement across recent positions
        movement = 0
        positions = track_history[track_id][-3:]  # Use only last 3 positions
        for i in range(1, len(positions)):
            # Calculate displacement in X and Y
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            # Add Euclidean distance
            movement += np.sqrt(dx*dx + dy*dy)
        
        # Return average movement
        if len(positions) <= 1:  # Prevent division by zero
            return 10
            
        return movement / (len(positions) - 1)
    except Exception as e:
        print(f"Error calculating movement for ID {track_id}: {e}")
        return 10  # Default value if error

# Function to save crossing statistics to file
def save_crossing_statistics():
    """Save all crossing statistics to a CSV file"""
    try:
        stats_path = os.path.join("stats", f"crossing_stats_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(stats_path, 'w') as f:
            # Write header
            f.write("ID,Direction,Start_Frame,End_Frame,Time_Seconds,Formatted_Time\n")
            
            # Write data for each crossing
            for crossing in all_crossings:
                f.write(f"{crossing['id']},{crossing['direction']},{crossing['start_frame']}," +
                       f"{crossing['end_frame']},{crossing['time_seconds']:.2f},{crossing['formatted_time']}\n")
                       
        print(f"Crossing statistics saved to {stats_path}")
        
        # Also save a summary
        summary_path = os.path.join("stats", f"crossing_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write("===== CROSSING TIME SUMMARY =====\n\n")
            
            # A to B statistics
            f.write("Zone A to Zone B Crossings:\n")
            f.write(f"Total crossings: {zone_a_to_b_count}\n")
            if crossing_times_a_to_b:
                avg_time = sum(crossing_times_a_to_b.values()) / len(crossing_times_a_to_b)
                min_time = min(crossing_times_a_to_b.values()) if crossing_times_a_to_b else 0
                max_time = max(crossing_times_a_to_b.values()) if crossing_times_a_to_b else 0
                f.write(f"Average time: {format_time(avg_time)}\n")
                f.write(f"Fastest time: {format_time(min_time)}\n")
                f.write(f"Slowest time: {format_time(max_time)}\n")
            f.write("\n")
            
            # B to A statistics
            f.write("Zone B to Zone A Crossings:\n")
            f.write(f"Total crossings: {zone_b_to_a_count}\n")
            if crossing_times_b_to_a:
                avg_time = sum(crossing_times_b_to_a.values()) / len(crossing_times_b_to_a)
                min_time = min(crossing_times_b_to_a.values()) if crossing_times_b_to_a else 0
                max_time = max(crossing_times_b_to_a.values()) if crossing_times_b_to_a else 0
                f.write(f"Average time: {format_time(avg_time)}\n")
                f.write(f"Fastest time: {format_time(min_time)}\n")
                f.write(f"Slowest time: {format_time(max_time)}\n")
            
        print(f"Crossing summary saved to {summary_path}")
        
    except Exception as e:
        print(f"Error saving crossing statistics: {e}")

# Setup de la ventana única para todo el proceso
print("\n===== DEFINICIÓN DE ZONAS =====")
print("Por favor, defina las zonas de detección:")
print("1. Zona de detección principal (NARANJA) - haga clic en 4 puntos")
print("2. Zona A (AZUL) - haga clic en 4 puntos")
print("3. Zona B (AZUL) - haga clic en 4 puntos")

try:
    # Create a single window for the entire application
    cv2.namedWindow(MAIN_WINDOW_NAME)
    cv2.setMouseCallback(MAIN_WINDOW_NAME, select_polygon)
    
    # Create stats window if enabled
    if SHOW_STATS_WINDOW:
        stats_window_name = "Estadísticas de Cruce"
        cv2.namedWindow(stats_window_name)
    
    # Step 1: Read first frame for polygon selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer fotograma")
        exit()
    
    # Reset video to beginning after getting first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Zone definition phase
    selection_frame = first_frame.copy()
    while in_zone_definition_mode:
        # Draw polygons and instructions on the frame
        selection_frame = first_frame.copy()
        draw_polygons(selection_frame)
        
        # Show the frame in the main window
        cv2.imshow(MAIN_WINDOW_NAME, selection_frame)
        
        # Check if all zones are defined
        if polygon_selection_done:
            # Wait for key press before continuing
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Any key pressed
                in_zone_definition_mode = False  # Exit zone definition mode
                break
        else:
            # Just wait briefly
            if cv2.waitKey(30) & 0xFF == 27:  # ESC to cancel
                print("Definición de zonas cancelada")
                exit()
    
    print("¡Definición de zonas completada!")
    
    # Warm up the GPU before starting real processing
    if GPU_ENABLED:
        print("Calentando GPU para optimizar rendimiento...")
        try:
            # Create dummy tensor for warm-up
            dummy_input = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)
            with torch.no_grad():
                for _ in range(5):  # Fewer iterations for GPUs with less memory
                    _ = model.predict(dummy_input, verbose=False)
            print("GPU calentada y lista para procesar")
            # Free warm-up memory
            del dummy_input
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error durante el calentamiento: {e}, continuando sin calentamiento...")

    print("\nIniciando procesamiento de video...")

    # Main processing loop - we continue using the same window
    while cap.isOpened():
        # Measure processing time per frame
        frame_start_time = time.time()
        
        # Read the next frame
        ret, frame = cap.read()
        if not ret:  # If no more frames
            print("Fin del video")
            break
        
        frame_count += 1  # Increment frame counter
        
        # Update visualization data age
        for i in range(len(visualization_data)):
            transition_type, start, end, frames = visualization_data[i]
            visualization_data[i] = (transition_type, start, end, frames + 1)
        
        # Remove old visualization data
        visualization_data = [item for item in visualization_data if item[3] < MAX_VISUALIZATION_TIME]
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100  # Percentage completed
            elapsed = time.time() - start_time  # Time elapsed
            # Estimate remaining time based on average time per frame
            remaining = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            
            # Calculate average FPS
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"Procesando... {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {avg_fps:.1f} - Tiempo restante: {remaining:.1f}s")
            
            # Show GPU memory usage if available
            if GPU_ENABLED:
                try:
                    used_mem = torch.cuda.memory_allocated() / 1e9
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"Memoria GPU: {used_mem:.2f}GB / {total_mem:.2f}GB ({used_mem/total_mem*100:.1f}%)")
                except:
                    print("No se pudo obtener información de memoria GPU")
        
        # Create a copy of the frame to draw results
        result_frame = frame.copy()
        
        # Draw all zone polygons on result frame
        draw_polygons(result_frame)
        
        # YOLO detection with tracking
        try:
            # Enhance frame for better detection
            enhanced_frame = enhance_frame(frame)
            
            # List to store results from multiple detections
            results_list = []
            
            # First detection pass with standard configuration
            try:
                if TRACK_PEOPLE_ONLY:
                    # Detect only people (class 0) with minimum confidence CONFIDENCE
                    # Use device parameter as appropriate
                    if GPU_ENABLED:
                        results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE, 
                                             iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS,
                                             device=device,
                                             imgsz=IMAGE_SIZE,  # Specify size to optimize GPU memory
                                             verbose=False)
                    else:
                        # If PyTorch doesn't detect CUDA but system has GPU, try to use GPU directly
                        if gpu_system_available:
                            # Direct attempt with device=0
                            try:
                                results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE, 
                                                   iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS,
                                                   device=0,  # Force use of first GPU
                                                   imgsz=IMAGE_SIZE,
                                                   verbose=False)
                            except:
                                # If it fails, use CPU
                                results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE, 
                                                   iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS,
                                                   device='cpu',
                                                   imgsz=IMAGE_SIZE,
                                                   verbose=False)
                        else:
                            # No GPU, use CPU
                            results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE, 
                                               iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS,
                                               device='cpu',
                                               imgsz=IMAGE_SIZE,
                                               verbose=False)
                            
                    if results and len(results) > 0:
                        results_list.append(results[0])
            except Exception as e:
                print(f"Error en primera detección: {e}")
            
            # Second pass with reduced confidence to capture more people
            try:
                # Similar to first pass but with adjusted confidence
                if GPU_ENABLED:
                    backup_results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE*0.8, 
                                               iou=IOU_THRESHOLD*1.2, max_det=MAX_DETECTIONS,
                                               device=device,
                                               imgsz=IMAGE_SIZE,
                                               verbose=False)
                else:
                    # Same handling as in first pass
                    if gpu_system_available:
                        try:
                            backup_results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE*0.8, 
                                                     iou=IOU_THRESHOLD*1.2, max_det=MAX_DETECTIONS,
                                                     device=0,
                                                     imgsz=IMAGE_SIZE,
                                                     verbose=False)
                        except:
                            backup_results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE*0.8, 
                                                     iou=IOU_THRESHOLD*1.2, max_det=MAX_DETECTIONS,
                                                     device='cpu',
                                                     imgsz=IMAGE_SIZE,
                                                     verbose=False)
                    else:
                        backup_results = model.track(enhanced_frame, persist=True, classes=[0], conf=CONFIDENCE*0.8, 
                                                 iou=IOU_THRESHOLD*1.2, max_det=MAX_DETECTIONS,
                                                 device='cpu',
                                                 imgsz=IMAGE_SIZE,
                                                 verbose=False)
                                                 
                if backup_results and len(backup_results) > 0:
                    results_list.append(backup_results[0])
            except Exception as e:
                print(f"Error en segunda detección: {e}")
                
            current_persons = 0  # Reset person counter for this frame
            # Set to avoid processing same object twice
            processed_boxes = set()
            
            # Process all detection results
            for result in results_list:
                # Verify result contains detection boxes
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                boxes = result.boxes
                
                # Extract tracking IDs with error handling
                track_ids = []
                if hasattr(boxes, 'id') and boxes.id is not None:
                    try:
                        # Convert tensor of IDs to list of integers
                        track_ids = boxes.id.int().cpu().tolist()
                    except Exception as e:
                        print(f"Error getting track_ids: {e}")
                        # If error, use indexes as IDs
                        track_ids = list(range(len(boxes)))
                else:
                    # If no tracking IDs, use indexes
                    track_ids = list(range(len(boxes)))
                
                # Process each detected box
                for i, box in enumerate(boxes):
                    try:
                        # Get tracking ID safely
                        track_id = i + 1000  # Default value with offset to avoid confusion
                        if track_ids and i < len(track_ids):
                            track_id = track_ids[i]
                        
                        # Avoid processing same object twice
                        box_id = f"{track_id}_{i}"
                        if box_id in processed_boxes:
                            continue
                        processed_boxes.add(box_id)
                        
                        # Extract bounding box coordinates with error handling
                        try:
                            if not hasattr(box, 'xyxy') or len(box.xyxy) == 0:
                                continue
                                
                            box_data = box.xyxy[0]  # xyxy format: [x1, y1, x2, y2]
                            if isinstance(box_data, torch.Tensor):
                                # Convert tensor to list of integers
                                x1, y1, x2, y2 = box_data.int().cpu().tolist()
                            else:
                                # If already a list, convert to integers
                                x1, y1, x2, y2 = [int(val) for val in box_data]
                        except Exception as e:
                            print(f"Error extracting box coordinates {i}: {e}")
                            continue
                        
                        # Extract class and confidence with error handling
                        try:
                            cls = 0  # Default value (person)
                            conf = 0.0  # Default confidence
                            
                            if hasattr(box, 'cls') and len(box.cls) > 0:
                                cls = int(box.cls[0])
                            
                            if hasattr(box, 'conf') and len(box.conf) > 0:
                                conf = float(box.conf[0])
                        except Exception as e:
                            print(f"Error extracting class/confidence for box {i}: {e}")
                            # Keep default values
                        
                        # Process only objects of class 'person' (class 0)
                        if cls == 0:  
                            # Verify proportions are compatible with a person
                            if not is_valid_human(x1, y1, x2, y2, track_id):
                                continue
                                
                            # Filter already identified static objects
                            if track_id in static_object_ids and track_id not in verified_person_ids:
                                continue
                            
                            # Calculate object center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Check if person is inside the main detection zone
                            if len(polygon_main) == 4:
                                if not point_in_polygon((center_x, center_y), polygon_main):
                                    continue  # Skip if person is outside detection zone
                            
                            # Record position in tracking history
                            track_history[track_id].append((center_x, center_y))
                            
                            # Determine zone using improved zone detection
                            current_zone = determine_zone(x1, y1, x2, y2, track_id)
                            
                            # Check if person has crossed between zones
                            zone_crossed, crossing_direction = check_zone_crossing(track_id, current_zone, x1, y1, x2, y2)
                            
                            # Movement analysis (only if enough history)
                            if len(track_history[track_id]) >= 10:
                                # Calculate recent average movement
                                avg_movement = calculate_movement(track_history, track_id)
                                
                                # Store velocity in registry
                                track_velocities[track_id] = avg_movement
                                
                                                                # Check if it's a static object (post, sign, etc.)
                                if avg_movement < MOVEMENT_THRESHOLD and track_id not in verified_person_ids:
                                    # Increment frame counter without movement
                                    static_object_counter = track_velocities.get(f"static_{track_id}", 0)
                                    track_velocities[f"static_{track_id}"] = static_object_counter + 1
                                    
                                    # Mark as static object after persistent lack of movement
                                    if track_velocities[f"static_{track_id}"] > 10:
                                        static_object_ids.add(track_id)
                                        continue
                                else:
                                    # If movement detected, reset static counter
                                    track_velocities[f"static_{track_id}"] = 0
                                    # If marked as static, rehabilitate it
                                    if track_id in static_object_ids:
                                        static_object_ids.remove(track_id)
                                        # Confirm as person if showing movement
                                        verified_person_ids.add(track_id)
                            
                            # Passed all checks: count as valid person
                            current_persons += 1
                            
                            # Get current stable zone for display
                            stable_zone = person_last_stable_zone.get(track_id, None)
                            
                            # Set color based on zone, crossing status, and transition
                            if zone_crossed:
                                # Just crossed - highlight with bright magenta
                                box_color = (255, 0, 255)  # Magenta
                            elif current_zone == 'a' or stable_zone == 'a':
                                # In zone A - blue
                                box_color = (255, 0, 0)  # Blue
                            elif current_zone == 'b' or stable_zone == 'b':
                                # In zone B - red  
                                box_color = (0, 0, 255)  # Red
                            elif current_zone == 'transition':
                                # In transition between zones
                                box_color = (0, 255, 255)  # Yellow
                            else:
                                # Default - green
                                box_color = (0, 255, 0)  # Green
                            
                            # Draw bounding box
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Add labels with enhanced info
                            zone_label = ""
                            if stable_zone:
                                zone_label += f" ZONE:{stable_zone.upper()}"
                            elif current_zone:
                                zone_label += f" ({current_zone.upper()})"
                                
                            # Add direction if just crossed
                            direction_label = ""
                            if crossing_direction == "a_to_b":
                                direction_label = " A→B"
                            elif crossing_direction == "b_to_a":
                                direction_label = " B→A"
                            
                            # NEW: Add crossing time if available
                            time_label = ""
                            if track_id in crossing_times_a_to_b:
                                time_label = f" [{format_time(crossing_times_a_to_b[track_id])}s]"
                            elif track_id in crossing_times_b_to_a:
                                time_label = f" [{format_time(crossing_times_b_to_a[track_id])}s]"
                                                      
                            label = f"ID:{track_id}{zone_label}{direction_label}{time_label} {conf:.2f}"
                            cv2.putText(result_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            
                            # Draw movement trail
                            trail_color = box_color  # Same color as the box
                            if len(track_history[track_id]) > 1:
                                # Limit trail length to 30 points
                                if len(track_history[track_id]) > 30:
                                    # Remove oldest points to keep only 30
                                    del track_history[track_id][0:len(track_history[track_id])-30]
                                
                                for j in range(1, len(track_history[track_id])):
                                    # Draw line between consecutive points
                                    pt1 = track_history[track_id][j - 1]
                                    pt2 = track_history[track_id][j]
                                    cv2.line(result_frame, pt1, pt2, trail_color, 2)
                    except Exception as e:
                        # If error processing a box, skip it and continue
                        print(f"Error processing box {i} in frame {frame_count}: {e}")
                        continue
            
            # Update smoothing system for people count
            # Remove oldest count and add current
            recent_persons.pop(0)
            recent_persons.append(current_persons)
            # Use recent maximum to stabilize count
            smooth_count = max(recent_persons)
            
        except Exception as e:
            # Capture any error in detection process
            print(f"Error in frame {frame_count}: {e}")
            smooth_count = current_persons  # Keep current count if error
        
        # Update maximum people counter
        if smooth_count > total_persons_max:
            total_persons_max = smooth_count
            
            # Save frame when a new maximum is detected
            try:
                max_path = os.path.join("detections", f"max_{total_persons_max}_people_frame_{frame_count}.jpg")
                cv2.imwrite(max_path, result_frame)
                print(f"New maximum! {total_persons_max} people detected in frame {frame_count}")
            except Exception as e:
                print(f"Error saving maximum image: {e}")
                
        # Add counters to result frame
        cv2.putText(result_frame, f"People: {smooth_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Calculate frame processing metrics
        frame_time = time.time() - frame_start_time
        processing_times.append(frame_time)
        fps_current = 1.0 / frame_time if frame_time > 0 else 0
        
        # Update and show statistics window
        if SHOW_STATS_WINDOW:
            stats_image = create_stats_window()
            cv2.imshow(stats_window_name, stats_image)
                
        # Show live preview in the same window we used for zone definition
        try:
            cv2.imshow(MAIN_WINDOW_NAME, result_frame)
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error showing preview: {e}")
            break
        
        # Save frame to video if enabled
        if SAVE_VIDEO and out is not None:
            try:
                out.write(result_frame)
            except Exception as e:
                print(f"Error saving frame to video: {e}")
                SAVE_VIDEO = False  # Disable if error
            
        # Save periodic captures for analysis
        if frame_count % 30 == 0:
            try:
                capture_path = os.path.join("detections", f"frame_{frame_count}.jpg")
                cv2.imwrite(capture_path, result_frame)
            except Exception as e:
                print(f"Error saving periodic capture: {e}")

        # Free GPU memory at end of each iteration (less frequent for better performance)
        if GPU_ENABLED and frame_count % 50 == 0:
            torch.cuda.empty_cache()

except Exception as e:
    # Capture general errors during processing
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()  # Show full stack trace

finally:
    # Save crossing statistics to file before exit
    save_crossing_statistics()
    
    # Free GPU memory
    if GPU_ENABLED:
        torch.cuda.empty_cache()
    
    # Close visualization windows
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Close output video file
    try:
        if SAVE_VIDEO and out is not None:
            out.release()
            print(f"\nVideo saved as: {output_path}")
    except:
        pass
    
    # Show final statistics
    total_time = time.time() - start_time
    fps_avg = frame_count / total_time if total_time > 0 else 0

    print("\n--- Final Results ---")
    print(f"Total frames processed: {frame_count}/{total_frames}")
    print(f"Maximum people detected: {total_persons_max}")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Average FPS: {fps_avg:.2f}")
    print(f"Zone transitions:")
    print(f"  Zone A to Zone B: {zone_a_to_b_count}")
    print(f"  Zone B to Zone A: {zone_b_to_a_count}")

    # Show time statistics
    print("\n--- Time Crossing Statistics ---")
    if crossing_times_a_to_b:
        avg_time = sum(crossing_times_a_to_b.values()) / len(crossing_times_a_to_b)
        min_time = min(crossing_times_a_to_b.values())
        max_time = max(crossing_times_a_to_b.values())
        print(f"A → B: Average: {format_time(avg_time)}s, MMin: {format_time(min_time)}s, MMax: {format_time(max_time)}s")

    if crossing_times_b_to_a:
        avg_time = sum(crossing_times_b_to_a.values()) / len(crossing_times_b_to_a)
        min_time = min(crossing_times_b_to_a.values())
        max_time = max(crossing_times_b_to_a.values())
        print(f"B → A: Average: {format_time(avg_time)}s, MMin: {format_time(min_time)}s, MMax: {format_time(max_time)}s")

    if len(processing_times) > 0:
        print(f"Average processing time per frame: {sum(processing_times)/len(processing_times)*1000:.1f} ms")

    # Show GPU information if available
    if GPU_ENABLED or gpu_system_available:
        print(f"Device used: GPU - NVIDIA GTX 1650Ti")
    else:
        print("Device used: CPU")

    # Free video resources
    try:
        if cap is not None:
            cap.release()
    except:
        pass

    print("Processing completed")
    print(f"Detailed statistics saved in 'stats' folder")
