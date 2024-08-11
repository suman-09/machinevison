import torch
import cv2
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Access the webcam
cap = cv2.VideoCapture(0)

# Variable to keep track of whether a cell phone was detected in the previous frame
cell_phone_was_detected = False

# Counter for cell phone detections
detection_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Perform object detection
    results = model(frame)
    
    # Get detections in pandas DataFrame format
    detected_objects = results.pandas().xyxy[0]
    
    # Flag to check if cell phone is detected in this frame
    cell_phone_found = False
    
    for index, row in detected_objects.iterrows():
        class_name = row['name']
        confidence = row['confidence']
        
        # Ignore detections of people
        if class_name.lower() == 'person':
            continue
        
        if class_name.lower() == 'cell phone':  # Change this to the desired class name
            cell_phone_found = True
            break  # Exit the detection loop

    # Print detection message and update count only if a cell phone is found and it wasn't detected in the previous frame
    if cell_phone_found and not cell_phone_was_detected:
        detection_count += 1
        print(f"Cell phone detected. Count: {detection_count}")
    
    # Update the cell phone detection state
    cell_phone_was_detected = cell_phone_found
    
    # Render and display the frame
    frame = results.render()[0]
    cv2.imshow('Webcam YOLOv5 Object Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
