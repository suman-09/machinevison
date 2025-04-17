import tornado.httpserver
import tornado.websocket
import tornado.concurrent
import tornado.ioloop
import tornado.web
import tornado.gen
import threading
import asyncio
import socket
import numpy as np
import imutils
import copy
import time
import cv2
import os
import warnings
import torch
import queue
from collections import deque

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Thread-safe frame queue
frame_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
lock = threading.Lock()
connectedDevices = set()

# Configuration parameters
CONFIDENCE_THRESHOLD = 0.20
PROCESS_EVERY_N_FRAMES = 7  # Process every 7th frame for detection
MAX_PROCESSING_FPS = 3     # Cap processing rate

# Buffering parameters
DISPLAY_BUFFER_SIZE = 8    # Number of frames to buffer before displaying
DISPLAY_DELAY = 4       # Target delay in seconds for the display buffer

# Detection smoothing parameters
SMOOTHING_WINDOW = 2       # Number of frames to average for smoothing
SMOOTHING_ALPHA = 0.3       # Alpha parameter for exponential moving average (0-1)
MAX_MISS_FRAMES = 10        # Keep showing last detection for this many frames if detection fails

# Load YOLOv5 model
print("Loading YOLOv5 model...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Configure model to detect only cell phones (class index 67 in COCO dataset)
    # Get all class names
    class_names = model.names
    
    # Find cell phone class index
    cell_phone_index = None
    for idx, name in class_names.items():
        if name.lower() == 'cell phone':
            cell_phone_index = idx
            break
    if cell_phone_index is not None:
        print(f"Cell phone class index: {cell_phone_index}")
        model.classes = [cell_phone_index]  # Only detect cell phones
    else:
        print("Warning: Cell phone class not found in model")
    model.conf = CONFIDENCE_THRESHOLD  # Set confidence threshold
    print("YOLOv5 model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv5 model: {str(e)}")
    model = None

# Frame processing thread
def process_frames_thread():
    last_process_time = 0
    while True:
        try:
            # Get frame from queue with a timeout to prevent blocking forever
            device, frame, timestamp = frame_queue.get(timeout=1.0)
            # Check if we should process this frame based on time
            current_time = time.time()
            min_interval = 1.0 / MAX_PROCESSING_FPS
            if current_time - last_process_time < min_interval:
                # Skip processing but still put the frame in display buffer
                with lock:
                    # Add the original frame to the display buffer
                    device.display_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': timestamp,
                        'processed': False
                    })
                    # Limit buffer size
                    while len(device.display_buffer) > DISPLAY_BUFFER_SIZE:
                        device.display_buffer.popleft()
                frame_queue.task_done()
                continue
                
            # Process frame
            if frame is not None and device.frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    # Process frame for object detection
                    frame_with_detection, servo_angle, detection_info = detect_mobile_phone(
                        frame.copy(), device.detection_history)
                    
                    # Update detection history
                    if detection_info is not None:
                        device.detection_history.append(detection_info)
                        while len(device.detection_history) > SMOOTHING_WINDOW:
                            device.detection_history.popleft()
                    
                    # If a phone is detected, send the servo angle command
                    if servo_angle is not None:
                        # Throttle commands to avoid overwhelming the ESP32
                        current_time = time.time()
                        if current_time - device.last_servo_command_time > 0.2:  # send at most 5 times per second
                            servo_command = f"servo:{servo_angle}"
                            print(f"Phone detected - Sending command: {servo_command}")
                            # Schedule the write_message call on the main event loop
                            tornado.ioloop.IOLoop.current().add_callback(
                                device.write_message, servo_command)
                            device.last_servo_command_time = current_time
                    
                    # Add the processed frame to the display buffer
                    with lock:
                        device.display_buffer.append({
                            'frame': frame_with_detection,
                            'timestamp': timestamp,
                            'processed': True,
                            'detection': detection_info is not None
                        })
                        # Limit buffer size
                        while len(device.display_buffer) > DISPLAY_BUFFER_SIZE:
                            device.display_buffer.popleft()
                    
                    # Encode the frame for streaming
                    (flag, encodedImage) = cv2.imencode(".jpg", frame_with_detection)
                    if flag:
                        device.outputFrame = encodedImage.tobytes()
                    
                    last_process_time = time.time()
                except Exception as e:
                    print(f"Error in object detection: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # For skipped frames, just add to display buffer without processing
                with lock:
                    device.display_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': timestamp,
                        'processed': False
                    })
                    # Limit buffer size
                    while len(device.display_buffer) > DISPLAY_BUFFER_SIZE:
                        device.display_buffer.popleft()
            
            # Mark task as done
            frame_queue.task_done()
        except queue.Empty:
            # Queue was empty, just continue
            pass
        except Exception as e:
            print(f"Error in processing thread: {str(e)}")
            # Don't exit the thread on error

# Start processing thread
processing_thread = threading.Thread(target=process_frames_thread, daemon=True)
processing_thread.start()

def detect_mobile_phone(frame, detection_history):
    # Skip detection if model couldn't be loaded
    if model is None:
        return frame, None, None
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    # Make a copy of frame for drawing
    display_frame = frame.copy()
    # Perform object detection with YOLOv5
    results = model(frame)
    # Get detections as a pandas DataFrame
    detected_objects = results.pandas().xyxy[0]
    # If no objects detected, return original frame
    if len(detected_objects) == 0:
        return display_frame, None, None
    # Find the best mobile phone detection (highest confidence)
    best_detection = None
    best_confidence = 0
    for _, row in detected_objects.iterrows():
        confidence = float(row['confidence'])
        # Only interested in cell phones with higher confidence
        if confidence > best_confidence:
            best_detection = row
            best_confidence = confidence
    # If we found a phone
    if best_detection is not None:
        # Get bounding box coordinates
        startX, startY = int(best_detection['xmin']), int(best_detection['ymin'])
        endX, endY = int(best_detection['xmax']), int(best_detection['ymax'])
        # Calculate the center X coordinate of the phone
        centerX = int((startX + endX) / 2)
        centerY = int((startY + endY) / 2)
        # Store current detection for smoothing
        current_detection = {
            'startX': startX, 'startY': startY,
            'endX': endX, 'endY': endY,
            'centerX': centerX, 'centerY': centerY,
            'confidence': best_confidence
        }
        # Apply temporal smoothing if we have previous detections
        if detection_history:
            # Calculate smoothed values using all detections in history
            smooth_startX = sum(d['startX'] for d in detection_history) / len(detection_history)
            smooth_startY = sum(d['startY'] for d in detection_history) / len(detection_history)
            smooth_endX = sum(d['endX'] for d in detection_history) / len(detection_history)
            smooth_endY = sum(d['endY'] for d in detection_history) / len(detection_history)
            smooth_centerX = sum(d['centerX'] for d in detection_history) / len(detection_history)
            # Weight current detection and history (70% history, 30% current)
            startX = int(0.7 * smooth_startX + 0.3 * startX)
            startY = int(0.7 * smooth_startY + 0.3 * startY)
            endX = int(0.7 * smooth_endX + 0.3 * endX)
            centerX = int(0.7 * smooth_centerX + 0.3 * centerX)
        # Calculate servo angle based on phone position (0-180 degrees)
        # Linear mapping from left edge (0) to right edge (180)
        raw_servo_angle = int((centerX / w) * 180)
        if raw_servo_angle > 180:
            raw_servo_angle = 180
        elif raw_servo_angle < 0:
            raw_servo_angle = 0
        # Add a bit more smoothing to servo angle
        servo_angle = raw_servo_angle
        if detection_history:
            # Calculate average servo angle from history using centerX values
            # Use full width for mapping instead of half-width
            historical_angles = [int((d['centerX'] / w) * 180) for d in detection_history]
            historical_angles = [min(max(angle, 0), 180) for angle in historical_angles]
            avg_angle = sum(historical_angles) / len(historical_angles)
            # Weighted blend (60% history, 40% current)
            servo_angle = int(0.6 * avg_angle + 0.4 * raw_servo_angle)
        # Draw smooth rectangle around phone
        cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # Add confidence text
        conf_text = f"Phone: {best_confidence:.2f}"
        cv2.putText(display_frame, conf_text, (startX, startY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Add crosshair at center of detection
        cv2.line(display_frame, (centerX - 10, centerY),
                 (centerX + 10, centerY), (0, 0, 255), 2)
        cv2.line(display_frame, (centerX, centerY - 10),
                 (centerX, centerY + 10), (0, 0, 255), 2)
        # Return the angle for servo positioning
        return display_frame, servo_angle, current_detection
    return display_frame, None, None

# Update display function to work with buffered frames
def display_frames():
    while True:
        with lock:  # Ensure thread-safe access to connectedDevices
            devices_copy = list(connectedDevices)
        
        for device in devices_copy:
            with lock:  # Ensure thread-safe access to device data
                # Only display frames if we have enough buffered
                if device.display_buffer and len(device.display_buffer) >= DISPLAY_BUFFER_SIZE * 0.7:
                    # Get the oldest frame from the buffer
                    frame_data = device.display_buffer.popleft()
                    frame_to_display = frame_data['frame'].copy()
                    
                    # Calculate actual buffer delay
                    current_time = time.time()
                    actual_delay = current_time - frame_data['timestamp']
                    
                    # Add FPS and buffer info to the frame
                    fps_text = f"FPS: {device.fps:.1f}"
                    buffer_text = f"Buffer: {len(device.display_buffer)}/{DISPLAY_BUFFER_SIZE}"
                    delay_text = f"Delay: {actual_delay:.2f}s"
                    processed_text = "Processed" if frame_data.get('processed', False) else "Raw"
                    
                    cv2.putText(frame_to_display, fps_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_to_display, buffer_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_to_display, delay_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_to_display, processed_text, (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Update device's displayed frame
                    device.frame = frame_to_display
                    
                    # Display the frame in a named window for each device
                    window_name = f"ESP32Cam-{device.id or 'unknown'}"
                    cv2.imshow(window_name, frame_to_display)
        
        # Wait for 1ms and check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Add small delay to control display rate
        time.sleep(0.03)  # ~30 FPS for display

# Start display thread
display_thread = threading.Thread(target=display_frames, daemon=True)
display_thread.start()

class WSHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        super(WSHandler, self).__init__(*args, **kwargs)
        self.outputFrame = None
        self.frame = None
        self.id = None
        self.executor = tornado.concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_servo_command_time = 0
        self.frame_counter = 0
        self.smooth_angle = 90  # Start with center position
        self.detection_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_frame_time = time.time()
        self.fps = 0
        self.frames_received = 0
        self.fps_update_time = time.time()
        self.processed_timestamp = 0
        self.missed_detections = 0  # Counter for consecutive missed detections
        self.last_detection = None  # Store the most recent detection
        self.display_buffer = deque(maxlen=DISPLAY_BUFFER_SIZE)  # Buffer for display frames

    def open(self):
        print('new connection')
        self.id = None
        connectedDevices.add(self)

    def on_message(self, message):
        if isinstance(message, bytes):
            current_time = time.time()
            self.frames_received += 1
            
            # Calculate FPS every second
            if current_time - self.fps_update_time >= 1.0:
                self.fps = self.frames_received / (current_time - self.fps_update_time)
                self.frames_received = 0
                self.fps_update_time = current_time
            
            # Decode frame
            frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # Skip adding to processing queue if buffer is already full enough
            with lock:
                buffer_fullness = len(self.display_buffer) / DISPLAY_BUFFER_SIZE
            
            # If buffer is less than 90% full, add frame to processing queue
            if buffer_fullness < 0.9:
                try:
                    # Add frame to processing queue with timestamp
                    frame_queue.put((self, frame, current_time), block=False)
                except queue.Full:
                    # If queue is full, skip this frame
                    pass
                    
            # Update frame counter
            self.frame_counter += 1
        else:
            if not message:
                message = "esp32cam"
            print(message)
            self.id = str(message)

    def on_close(self):
        print('connection closed')
        connectedDevices.remove(self)

    def check_origin(self, origin):
        return True

application = tornado.web.Application([
    (r'/ws', WSHandler),
])

if __name__ == "__main__":
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at %s ***' % myIP)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(3000)
    tornado.ioloop.IOLoop.current().start()