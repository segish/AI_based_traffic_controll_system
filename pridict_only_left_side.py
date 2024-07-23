import os
import cv2
import numpy as np
from ultralytics import YOLO

# Path to the input video and output directory
VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'los_angeles.mp4')
output_image_dir = './output_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

# Load the model
model_path = 'yolov8x.pt'  # Path to the YOLO model weights
model = YOLO(model_path)

# Open the input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Calculate the interval between frames to extract
num_images_to_save = 10
frame_interval = total_frames // num_images_to_save

# Detection threshold
threshold = 0.5

# Function to process and save a frame
def process_and_save_frame(frame, frame_idx):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Initialize variables for the dividing line
    dividing_line = None
    max_length = 0

    # Iterate through detected lines to find the longest one (assuming it is the dividing line)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > max_length:
                    max_length = length
                    dividing_line = (x1, y1, x2, y2)

    # Draw the dividing line on the frame
    if dividing_line is not None:
        x1, y1, x2, y2 = dividing_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Define a mask for the left-hand side of the dividing line
    mask = np.zeros((H, W), dtype=np.uint8)
    if dividing_line is not None:
        cv2.fillPoly(mask, [np.array([[0, H], [0, 0], [x1, y1], [x2, y2], [W, 0], [W, H]])], 255)

    # Detect cars in the frame
    results = model(frame)[0]

    # Initialize car count
    car_count = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Check if the center of the bounding box is within the left-hand side of the dividing line
            box_center_x = int((x1 + x2) / 2)
            box_center_y = int((y1 + y2) / 2)

            if mask[box_center_y, box_center_x] == 255:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                text = results.names[int(class_id)].upper()
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)

                cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Increment car count
                car_count += 1

    # Print the number of cars detected on the left-hand side
    cv2.putText(frame, f'Cars detected: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the annotated frame
    output_image_path = os.path.join(output_image_dir, f'frame_{frame_idx}.jpg')
    cv2.imwrite(output_image_path, frame)
    print(f"Saved frame {frame_idx} as '{output_image_path}'")

# Process and save frames at the calculated intervals
frame_idx = 0
while frame_idx < total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break
    
    process_and_save_frame(frame, frame_idx)
    frame_idx += frame_interval

# Release the video capture
cap.release()
cv2.destroyAllWindows()
