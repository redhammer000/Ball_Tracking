import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = r'C:\Users\fahee\Desktop\New Ball\batsmanbat.pt'
model = YOLO(model_path)

# Video path and output settings
video_path = r'C:\Users\fahee\Desktop\New Ball\LBW4.mp4'
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
stump_positions = []
frame_counter = 0
refresh_interval = 15
trajectory_length = 50
fade_strength = 0.8
num_future_frames = 10

# Class IDs
BALL_CLASS = 0
STUMP_CLASS = 1
BATSMAN_CLASS = 2

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                   [0, 1, 0, 1],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# Main loop to process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, conf=0.4)

    ball_position = None
    detected_ball = False
    batsman_mask = np.zeros_like(frame, dtype=np.uint8)  # Ensure it's uint8 type

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            label = int(box.cls[0])
            x1, y1, x2, y2 = map(int, coords)

            color = (255, 0, 0)  # Default color for bounding boxes
            if label == BALL_CLASS:
                color = (0, 0, 255)  # Red for ball's circle
                detected_ball = True
                ball_position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, ball_position, 5, color, -1)

                # Kalman Filter update
                kalman.correct(np.array(ball_position, dtype=np.float32))
                predicted = kalman.predict()

                # Predict future positions
                future_positions = []
                for i in range(1, num_future_frames + 1):
                    kalman.transitionMatrix[0, 2] = i
                    kalman.transitionMatrix[1, 3] = i
                    future_predicted = kalman.predict()
                    future_position = (int(future_predicted[0]), int(future_predicted[1]))
                    future_positions.append(future_position)

                # Update trajectory with recent predictions
                if len(future_positions) > 0:
                    trajectory = trajectory[-(trajectory_length-1):] + future_positions
                else:
                    trajectory = trajectory[-(trajectory_length-1):]

            elif label == STUMP_CLASS:
                color = (255, 0, 0)  # Blue for stumps
                stump_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                stump_positions.append(stump_position)

            elif label == BATSMAN_CLASS:
                batsman_mask[y1:y2, x1:x2] = 255  # Create mask for the batsman

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Apply opacity reduction to batsman area
    batsman_mask = batsman_mask.astype(np.float32) / 255.0  # Normalize mask to [0, 1] range
    frame = frame.astype(np.float32)
    frame = cv2.addWeighted(frame, 1.0, frame * batsman_mask * -0.1, 0.5, 0).astype(np.uint8)

    # Refresh trajectory tracking periodically
    if frame_counter % refresh_interval == 0:
        if not detected_ball:
            trajectory = []

    # Draw the trajectory of the ball with a trail effect
    for i in range(len(trajectory) - 1):
        alpha = min(1.0, (1.0 - (i / len(trajectory)) * fade_strength))
        color = (0, 0, 255)
        cv2.line(frame, trajectory[i], trajectory[i + 1], color, 2, lineType=cv2.LINE_AA)

    # Draw a vertical line down to the next stump position only if more than one stump is detected
    if len(stump_positions) > 1:
        # Draw a straight line between the top and bottom stumps
        top_stump = min(stump_positions, key=lambda pos: pos[1])
        bottom_stump = max(stump_positions, key=lambda pos: pos[1])
        cv2.line(frame, top_stump, bottom_stump, (255, 255, 255), 3)

    # Save the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Object Detection and Trajectory', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
