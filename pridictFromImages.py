import os
from ultralytics import YOLO
import cv2

# Path to the input image
image_path = './images/input_images/imhh.png'  
output_image_path = './images/output_images/imhh.png'  

# Load the model
model_path = 'yolov8x.pt'  # Path to the YOLO model weights
model = YOLO(model_path)

# Read the input image
frame = cv2.imread(image_path)

# Detection threshold
threshold = 0

# Detect cars in the image
results = model(frame)[0]

# Initialize car count
car_count = 0

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Draw bounding box and label# Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        font_scale = 0.5  # Smaller font size
        text = results.names[int(class_id)].upper()
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + text_width, int(y1) - text_height - baseline), (0, 255, 0), -1)
        cv2.putText(frame, text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        car_count += 1

# Print the number of cars detected on the image
cv2.putText(frame, f'{car_count} vehicles', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

# Save the annotated image
cv2.imwrite(output_image_path, frame)

print(f"Annotated image saved as '{output_image_path}'")
