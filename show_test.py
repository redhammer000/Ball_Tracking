import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = r'C:\Users\fahee\Desktop\New Ball\bestbat.pt'
model = YOLO(model_path)

# Video path and output settings
video_path = r'C:\Users\fahee\Desktop\New Ball\LBW5.mp4'
output_path = r'C:\Users\fahee\Desktop\New Ball\output.mp4'

# Open video capture
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize variables
trajectory = []
trajectory_length = 20  # Keep more points to create a smoother trail
imgsz = 640  # Image size used during training

# Class IDs
BALL_CLASS = 0

# Initialize Kalman Filter
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

# Main loop to process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match the training input size
    resized_frame = cv2.resize(frame, (imgsz, imgsz))

    # Perform object detection on the resized frame
    results = model(resized_frame, conf=0.35)

    ball_position = None

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            label = int(box.cls[0])
            x1, y1, x2, y2 = map(int, coords)

            if label == BALL_CLASS:
                # Convert coordinates back to the original frame size
                x1 = int(x1 * (width / imgsz))
                y1 = int(y1 * (height / imgsz))
                x2 = int(x2 * (width / imgsz))
                y2 = int(y2 * (height / imgsz))

                ball_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, ball_position, 5, (0, 0, 255), -1)

                # Update the Kalman filter with the detected position
                kalman.correct(np.array(ball_position, dtype=np.float32))
                predicted = kalman.predict()

                # Add predicted position to trajectory
                trajectory.append((int(predicted[0]), int(predicted[1])))

                # Maintain only the most recent points in the trajectory
                trajectory = trajectory[-trajectory_length:]

    # Draw the trajectory with a trail effect
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Save the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Object Detection and Trajectory', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
