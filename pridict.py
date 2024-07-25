import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

cap = cv2.VideoCapture('./videos/lane1.mp4')
ret, frame = cap.read()

model_path = os.path.join('yolov8x.pt')

model = YOLO("yolov8x.pt")

threshold = 0.3

while ret:
    car_count = 0 
    results = model(frame)[0]
    

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            car_count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            text = results.names[int(class_id)].upper()
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)

            cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Cars detected: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame",frame) 
    if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    # out.write(frame)
    ret, frame = cap.read()

cap.release()
# out.release()
cv2.destroyAllWindows()