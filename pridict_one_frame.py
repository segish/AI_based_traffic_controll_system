import os
from ultralytics import YOLO
import cv2
import time
import serial

portVar = "COM9"

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

laneNumber = 0
for video_file in video_files:
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


    threshold = 0.3
    results = model(frame)[0]

    vehicle_count = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and class_id in vehicle_class_ids:

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            text = results.names[int(class_id)].upper()
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)

            cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            vehicle_count += 1

    cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_image_path, frame)
    laneNumber += 1

    print(f"Annotated image saved as '{output_image_path}'")

    countdown_start = 2 * vehicle_count
    if(laneNumber == 1):
        serialInst.write(("lane1," + str(countdown_start)).encode('utf-8'))
    elif(laneNumber == 2):
        serialInst.write(("lane2," + str(countdown_start)).encode('utf-8'))
    elif(laneNumber == 3):
        serialInst.write(("lane3," + str(countdown_start)).encode('utf-8'))
    elif(laneNumber == 4):
        serialInst.write(("lane4," + str(countdown_start)).encode('utf-8'))

    # cv2.imshow('lane ' + str(laneNumber), frame)
    for i in range(countdown_start, -1, -1):
        print(i)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    # cv2.destroyAllWindows()
    cap.release()

serialInst.close()
print("All videos processed and program terminated")
