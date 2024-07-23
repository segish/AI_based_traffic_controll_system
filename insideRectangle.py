import os
from ultralytics import YOLO
import cv2
import numpy as np
import time
import serial

# portVar = "COM9"

# try:
#     serialInst = serial.Serial(portVar, 9600)
#     serialInst.timeout = 1
#     serialInst.flushInput()  # Clear input buffer
# except serial.SerialException as e:
#     print(f"Failed to open serial port '{portVar}': {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")

# Paths
VIDEOS_DIR = os.path.join('.', 'videos')
output_dir = 'output_images'  # Directory to save the annotated images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the model
model_path = 'bestbest.pt'
model = YOLO(model_path)

# Get list of video files in the directory
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]

# Vehicle-related class IDs
vehicle_class_ids = [2, 3, 5, 7]

# Define the rectangle coordinates
rect_points = np.array([(13, 355), (604, 356), (364, 4), (141, 13)], np.int32)
rect_points = rect_points.reshape((-1, 1, 2))

def is_point_in_polygon(polygon, point):
    # Check if a point is inside a polygon using cv2.pointPolygonTest
    return cv2.pointPolygonTest(polygon, point, False) >= 0

laneNumber = 0
# Process each video file
for video_file in video_files:
    if laneNumber>4:
        laneNumber=0
    video_path = os.path.join(VIDEOS_DIR, video_file)
    output_image_path = os.path.join(output_dir, f'{os.path.splitext(video_file)[0]}_detection.jpg')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    # Move to the last frame 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read the last frame from the video {video_path}")
        cap.release()
        continue

    cv2.polylines(frame, [np.array(rect_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    threshold = 0.3
    results = model(frame)[0]

    # Initialize vehicle count
    vehicle_count = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and class_id in vehicle_class_ids:
            # Check if the bounding box is inside the defined rectangle
            box_points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
            box_points = box_points.reshape((-1, 1, 2))

            # Check if any corner of the bounding box is inside the rectangle
            if any(is_point_in_polygon(rect_points, (int(x), int(y))) for x, y in box_points.reshape(-1, 2)):
                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                   # Smaller font size
                 text = results.names[int(class_id)].upper()
                 (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Draw the label background rectangle
                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)

                    # Put the text inside the rectangle
                 cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                 vehicle_count += 1

    cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the annotated image
    cv2.imshow("output_image_path", frame)
    laneNumber += 1

    print(f"Annotated image saved as '{output_image_path}'")

    # Perform countdown from 2 times the number of detected vehicles down to zero
    countdown_start = 2 * vehicle_count
    # if laneNumber == 1:
    #     serialInst.write(("lane1," + str(countdown_start)).encode('utf-8'))
    # elif laneNumber == 2:
    #     serialInst.write(("lane2," + str(countdown_start)).encode('utf-8'))
    # elif laneNumber == 3:
    #     serialInst.write(("lane3," + str(countdown_start)).encode('utf-8'))
    # elif laneNumber == 4:
    #     serialInst.write(("lane4," + str(countdown_start)).encode('utf-8'))

    for i in range(countdown_start, -1, -1):
        print(i)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cap.release()

# serialInst.close() 
print("All videos processed and program terminated")
