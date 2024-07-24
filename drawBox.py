import cv2
import numpy as np

# Global variable to store the points
points = []

def click_event(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select Rectangle', frame)

# Open the video file
video_path = './videos/lane2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Go to the last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

# Read the last frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

cv2.imshow('Select Rectangle', frame)
cv2.setMouseCallback('Select Rectangle', click_event)

# Wait until user selects four points
print("Click on the video frame to select the four coordinates of the rectangle.")
cv2.waitKey(0)

if len(points) != 4:
    print("Error: You need to select exactly four points.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Calculate the bounding box coordinates
x1, y1 = points[0]
x2, y2 = points[1]
x3, y3 = points[2]
x4, y4 = points[3]
print(points[0])
print(points[1])
print(points[2])
print(points[3])
# Draw the rectangle
rect_points = [points[0], points[1], points[3], points[2], points[0]]
rect_points = [(x, y) for x, y in rect_points]
rect_points = [(int(x), int(y)) for x, y in rect_points]

# Reset the video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the rectangle on the frame
    cv2.polylines(frame, [np.array(rect_points)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the frame
    cv2.imshow('Video with Rectangle', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
