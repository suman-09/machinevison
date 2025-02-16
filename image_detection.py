import torch
import cv2
import warnings
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading, time

# Configuration
laptop_ip = '172.16.146.177'
pico_ip = '172.16.146.202'  # Replace with your Pico W's IP address

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/clockwise':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Motor Clockwise')
        elif self.path == '/counterclockwise':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Motor Counterclockwise')
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('0.0.0.0', 5500)
    httpd = HTTPServer(server_address, MyHandler)
    print("Running Server on port 5500")
    httpd.serve_forever()

# Start the HTTP server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Access the webcam
cap = cv2.VideoCapture(0)

detection_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Show live feed
    cv2.imshow('Press Space to Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Spacebar to capture image
        print("Image captured, processing...")
        
        # Perform object detection
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]
        
        cell_phone_found = False
        
        for _, row in detected_objects.iterrows():
            class_name = row['name'].lower()
            if class_name == 'cell phone':
                cell_phone_found = True
                x_centre = (row['xmin'] + row['xmax']) / 2
                try:
                    if x_centre > frame.shape[1] / 2:
                        response = requests.get(f'http://{pico_ip}:5500/clockwise')
                        print(f"Sending clockwise GET request: {response.status_code}")
                    else:
                        response = requests.get(f'http://{pico_ip}:5500/counterclockwise')
                        print(f"Sending counterclockwise GET request: {response.status_code}")
                except requests.RequestException as e:
                    print(f"Failed to send request: {e}")
                break
        
        if cell_phone_found:
            detection_count += 1
            print(f"Cell phone detected. Count: {detection_count}")
        else:
            print("No cell phone found.")
        
        # Show the processed image with detections
        frame = results.render()[0]
        cv2.imshow('Captured Image with Detections', frame)
    
    elif key == ord('q'):  # Quit on 'q'
        break

cap.release()
cv2.destroyAllWindows()
