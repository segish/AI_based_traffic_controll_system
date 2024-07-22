import os
from ultralytics import YOLO
import cv2
import numpy as np
import serial

portVar = "COM2"

try:
    serialInst = serial.Serial(portVar, 9600)
    serialInst.timeout = 1
    serialInst.flushInput() 

except serial.SerialException as e:
    print(f"Failed to open serial port '{portVar}': {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Paths
VIDEOS_DIR = os.path.join('.', 'videos')
output_dir = 'output_images'  # Directory to save the annotated images

os.makedirs(output_dir, exist_ok=True)

model_path = 'yolov8x.pt'
model = YOLO(model_path)

video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')]

vehicle_class_ids = [2, 3, 5, 7]
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
weights = {
    2: 1.0,   # car
    3: 0.5,   # Motorcycle
    5: 1.5,   # Bus
    7: 2.0    # Truck
}
rect_points = np.array([(700, 401), (50, 1200), (2207, 1200), (950, 403)], np.int32) #for los_angeles.mp4 
rect_points_todraw = [(x, y) for x, y in rect_points]
rect_points_todraw = [(int(x), int(y)) for x, y in rect_points_todraw]
rect_points = rect_points.reshape((-1, 1, 2))

def is_point_in_polygon(polygon, point):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

laneNumber = 0
for video_file in video_files:
    if laneNumber>4:
        laneNumber=0
    video_path = os.path.join(VIDEOS_DIR, video_file)
    output_image_path = os.path.join(output_dir, f'{os.path.splitext(video_file)[0]}_detection.jpg')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read the last frame from the video {video_path}")
        cap.release()
        continue
    # draw the deviding rectangle
    cv2.polylines(frame, [np.array(rect_points)], isClosed=True, color=(0, 255, 0), thickness=2)

    threshold = 0.3
    results = model(frame)[0]

    # Initialize vehicle counts
    vehicle_counts = {class_id: 0 for class_id in vehicle_class_ids}

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and class_id in vehicle_class_ids:
            box_points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.int32)
            box_points = box_points.reshape((-1, 1, 2))

            # Check if any corner of the bounding box is inside the rectangle
            if any(is_point_in_polygon(rect_points, (int(x), int(y))) for x, y in box_points.reshape(-1, 2)):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                text = results.names[int(class_id)].upper()
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)

                cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                vehicle_counts[class_id] += 1

    # Display vehicle counts on the image
    y_offset = 50
    for class_id in vehicle_class_ids:
        class_name = class_list[class_id]
        count = vehicle_counts[class_id]
        cv2.putText(frame, f'{class_name}: {count}', (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 50

    cv2.imwrite(output_image_path, frame)
    cv2.imshow("count", frame)
    laneNumber += 1

    print(f"Annotated image saved as '{output_image_path}'")

    # countdown_start = 2 * sum(vehicle_counts.values())
    countdown_start = (
    weights.get(7, 0) * vehicle_counts.get(7, 0) +  # Truck
    weights.get(5, 0) * vehicle_counts.get(5, 0) +  # Bus
    weights.get(3, 0) * vehicle_counts.get(3, 0) +  # Car
    weights.get(2, 0) * vehicle_counts.get(2, 0)    # Motorcycle
)
    countdown_start= round(countdown_start)
    if countdown_start<=5: countdown_start = 5
    if laneNumber == 1:
        serialInst.write(("lane1," + str(countdown_start)).encode('utf-8'))
    elif laneNumber == 2:
        serialInst.write(("lane2," + str(countdown_start)).encode('utf-8'))
    elif laneNumber == 3:
        serialInst.write(("lane3," + str(countdown_start)).encode('utf-8'))
    elif laneNumber == 4:
        serialInst.write(("lane4," + str(countdown_start)).encode('utf-8'))

    for i in range(countdown_start, -1, -1):
        print(i)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

serialInst.close()
print("All videos processed and program terminated")
