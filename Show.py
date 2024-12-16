import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------
# Step 1: Load the YOLOv8 Model
# -----------------------------------
model_path = r'C:\Users\fahee\Desktop\New Ball\batsmanbat.pt'
model = YOLO(model_path)

# -----------------------------------
# Step 2: Set Video Paths and Output Settings
# -----------------------------------
video_path = r'C:\Users\fahee\Desktop\New Ball\LBW5.mp4'
output_path = r'C:\Users\fahee\Desktop\New Ball\output.mp4'

# -----------------------------------
# Step 3: Initialize Video Capture
# -----------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------------
# Step 4: Set Up Video Writer
# -----------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------------
# Step 5: Initialize Tracking Variables
# -----------------------------------
trajectory = []
trajectory_length = 20  # Number of points to keep in the trajectory
imgsz = 640  # Image size used during training

# -----------------------------------
# Step 6: Dynamically Map Class IDs
# -----------------------------------
class_names = model.names  # Dictionary: {class_id: class_name}

# Initialize class ID variables
BALL_CLASS = None
STUMP_CLASS = None
BATSMAN_CLASS = None

# Map class names to their corresponding IDs
for class_id, class_name in class_names.items():
    if class_name.lower() == 'ball':
        BALL_CLASS = class_id
    elif class_name.lower() == 'stump':
        STUMP_CLASS = class_id
    elif class_name.lower() == 'batsman':
        BATSMAN_CLASS = class_id

# Verify that all required classes are found
if BALL_CLASS is None or STUMP_CLASS is None or BATSMAN_CLASS is None:
    print("Error: One or more class names not found in the model.")
    print(f"Available classes: {class_names}")
    exit()

print(f"Class IDs - Ball: {BALL_CLASS}, Stump: {STUMP_CLASS}, Batsman: {BATSMAN_CLASS}")

# -----------------------------------
# Step 7: Set Detection Confidence Thresholds
# -----------------------------------
ball_conf_thresh = 0.35       # Higher threshold for ball detection
batsman_conf_thresh = 0.25    # Lower threshold for batsman detection
stump_conf_thresh = 0.30      # Threshold for stump detection

# -----------------------------------
# Step 8: Initialize Kalman Filter for Ball Tracking
# -----------------------------------
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

# -----------------------------------
# Step 9: Main Processing Loop
# -----------------------------------
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Resize frame to match the training input size
    resized_frame = cv2.resize(frame, (imgsz, imgsz))

    # Perform object detection on the resized frame
    results = model(resized_frame)

    # Initialize positions
    ball_position = None
    batsman_position = None
    stump_position = None

    # Process detections
    for result in results:
        if not result.boxes:
            continue  # No detections in this frame
        for box in result.boxes:
            # Extract bounding box coordinates, label, and confidence
            coords = box.xyxy[0].cpu().numpy()
            label = int(box.cls[0])
            confidence = box.conf[0]

            x1, y1, x2, y2 = map(int, coords)

            # Scale coordinates back to original frame size
            x_scale = width / imgsz
            y_scale = height / imgsz
            x1_orig = max(0, int(x1 * x_scale))
            y1_orig = max(0, int(y1 * y_scale))
            x2_orig = min(width, int(x2 * x_scale))
            y2_orig = min(height, int(y2 * y_scale))

            # Get the class name for debugging
            detected_class = class_names.get(label, "Unknown")
            print(f"Detected {detected_class} with confidence {confidence:.2f} at ({x1_orig}, {y1_orig}), ({x2_orig}, {y2_orig})")

            # -----------------------------------
            # Ball Detection
            # -----------------------------------
            if label == BALL_CLASS and confidence > ball_conf_thresh:
                ball_position = (int((x1_orig + x2_orig) / 2), int((y1_orig + y2_orig) / 2))
                cv2.circle(frame, ball_position, 5, (0, 0, 255), -1)

                # Update Kalman Filter with the detected position
                kalman.correct(np.array([ball_position[0], ball_position[1]], dtype=np.float32))
                predicted = kalman.predict()

                # Add predicted position to trajectory
                trajectory.append((int(predicted[0]), int(predicted[1])))

                # Maintain only the most recent points in the trajectory
                trajectory = trajectory[-trajectory_length:]

            # -----------------------------------
            # Batsman Detection and Transparency
            # -----------------------------------
            elif label == BATSMAN_CLASS and confidence > batsman_conf_thresh:
                batsman_position = (x1_orig, y1_orig, x2_orig, y2_orig)
                print(f"Batsman detected in frame {frame_count} at {batsman_position}")

                # Define the ROI (Region of Interest)
                roi = frame[y1_orig:y2_orig, x1_orig:x2_orig]

                # Check if ROI is valid
                if roi.size == 0:
                    print(f"Invalid ROI for batsman in frame {frame_count}. Skipping transparency.")
                    continue

                # Define the transparency factor
                alpha = 0.3  # 0.0 fully transparent through 1.0 fully opaque

                # Create a semi-transparent overlay by blending the ROI with a solid color
                overlay = frame.copy()
                # Define the color for the overlay (e.g., red)
                overlay_color = (0, 0, 255)  # Red in BGR

                # Draw a filled rectangle over the batsman region on the overlay
                cv2.rectangle(overlay, (x1_orig, y1_orig), (x2_orig, y2_orig), overlay_color, -1)

                # Blend the overlay with the original frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Optionally draw a bounding box around the batsman
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)

            # -----------------------------------
            # Stump Detection
            # -----------------------------------
            elif label == STUMP_CLASS and confidence > stump_conf_thresh:
                stump_position = (x1_orig, y1_orig, x2_orig, y2_orig)
                cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 255), 2)

    # -----------------------------------
    # Step 10: Draw Ball Trajectory
    # -----------------------------------
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # -----------------------------------
    # Step 11: Write and Display Frame
    # -----------------------------------
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Object Detection and Trajectory', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

# -----------------------------------
# Step 12: Release Resources
# -----------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
