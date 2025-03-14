import cv2 as cv
import numpy as np
from collections import deque
import os

AVG = 30
video_path = "input/incline/l1.mp4"  # Change this to your video path; if "flat" is in the path, display bottom 50%
clip = cv.VideoCapture(video_path)

bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
kernel = np.ones((5, 5), np.uint8)

# Store head and shoulder positions
head_path = deque(maxlen=100)          # For drawing the head path
shoulder_path = deque(maxlen=100)      # For drawing the shoulder path
head_positions = deque(maxlen=60)      # For calculating the mean head position (last 60 frames)
shoulder_positions = deque(maxlen=60)  # For calculating the mean shoulder position (last 60 frames)
mean_head_positions = []               # To store the computed mean head positions over time
mean_shoulder_positions = []           # To store the computed mean shoulder positions over time

# Variables for saving output video
writer = None
fps = clip.get(cv.CAP_PROP_FPS)

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = clip.read()
    if not ret:
        break

    frame_copy = frame.copy()

    # Apply background subtraction to find moving objects
    fgmask = bg_subtractor.apply(frame)
    _, thresh = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)

    # Clean up the mask
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # Find contours of moving objects
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    head_found = False
    shoulder_found = False

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Process only the largest contour (assuming it's the person)
    if contours and cv.contourArea(contours[0]) > 1000:  # Minimum area to be considered a person
        contour = contours[0]
        x, y, w, h = cv.boundingRect(contour)

        # Draw bounding box around the person
        cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the top 10% of the bounding box (head region)
        head_region_y = y
        head_region_h = int(h * 0.15)
        head_region = thresh[head_region_y:head_region_y + head_region_h, x:x + w]

        # Get the 10-20% of the bounding box (shoulder region)
        shoulder_region_y = y + head_region_h
        shoulder_region_h = int(h * 0.15)  # 10% for the shoulder region
        shoulder_region = thresh[shoulder_region_y:shoulder_region_y + shoulder_region_h, x:x + w]

        # Draw the head region
        cv.rectangle(frame_copy, (x, head_region_y), (x + w, head_region_y + head_region_h), (255, 0, 0), 2)
        # Draw the shoulder region
        cv.rectangle(frame_copy, (x, shoulder_region_y), (x + w, shoulder_region_y + shoulder_region_h), (0, 0, 255), 2)

        # Detect the head
        if head_region.size > 0:
            head_contours, _ = cv.findContours(head_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if head_contours:
                head_contour = max(head_contours, key=cv.contourArea)
                if cv.contourArea(head_contour) > 50:  # Minimum size for head
                    M = cv.moments(head_contour)
                    if M["m00"] != 0:
                        head_x = int(M["m10"] / M["m00"]) + x
                        head_y = int(M["m01"] / M["m00"]) + head_region_y
                        head_position = (head_x, head_y)
                        head_found = True

        # Detect the shoulder
        if shoulder_region.size > 0:
            shoulder_contours, _ = cv.findContours(shoulder_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if shoulder_contours:
                shoulder_contour = max(shoulder_contours, key=cv.contourArea)
                if cv.contourArea(shoulder_contour) > 50:  # Minimum size for shoulder
                    M = cv.moments(shoulder_contour)
                    if M["m00"] != 0:
                        shoulder_x = int(M["m10"] / M["m00"]) + x
                        shoulder_y = int(M["m01"] / M["m00"]) + shoulder_region_y
                        shoulder_position = (shoulder_x, shoulder_y)
                        shoulder_found = True

        # If no good contour found for head, use center of upper part
        if not head_found:
            head_position = (x + w // 2, y + head_region_h // 2)
            head_found = True

        # If no good contour found for shoulder, use center of shoulder region
        if not shoulder_found:
            shoulder_position = (x + w // 2, shoulder_region_y + shoulder_region_h // 2)
            shoulder_found = True

        # Add positions to lists
        if head_found:
            head_path.append(head_position)
            head_positions.append(head_position)
        if shoulder_found:
            shoulder_path.append(shoulder_position)
            shoulder_positions.append(shoulder_position)

        # Draw head and shoulder
        if head_found:
            cv.circle(frame_copy, head_position, 10, (0, 0, 255), 2)
            cv.putText(frame_copy, "Head", (head_position[0] + 10, head_position[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if shoulder_found:
            cv.circle(frame_copy, shoulder_position, 10, (0, 255, 0), 2)
            cv.putText(frame_copy, "Shoulder", (shoulder_position[0] + 10, shoulder_position[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw paths for head and shoulder
    for i in range(1, len(head_path)):
        if head_path[i - 1] and head_path[i]:
            cv.line(frame_copy, head_path[i - 1], head_path[i], (255, 0, 0), 2)
    for i in range(1, len(shoulder_path)):
        if shoulder_path[i - 1] and shoulder_path[i]:
            cv.line(frame_copy, shoulder_path[i - 1], shoulder_path[i], (0, 255, 0), 2)

    # Calculate mean positions for head and shoulder using the last 60 frames
    if len(head_positions) > 5:
        xs = [p[0] for p in head_positions]
        ys = [p[1] for p in head_positions]
        mean_head = (int(np.mean(xs)), int(np.mean(ys)))
        mean_head_positions.append(mean_head)
        cv.circle(frame_copy, mean_head, 5, (0, 255, 255), -1)
        cv.putText(frame_copy, "Mean Head", (mean_head[0] + 10, mean_head[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if len(shoulder_positions) > 5:
        xs = [p[0] for p in shoulder_positions]
        ys = [p[1] for p in shoulder_positions]
        mean_shoulder = (int(np.mean(xs)), int(np.mean(ys)))
        mean_shoulder_positions.append(mean_shoulder)
        cv.circle(frame_copy, mean_shoulder, 5, (255, 255, 0), -1)
        cv.putText(frame_copy, "Mean Shoulder", (mean_shoulder[0] + 10, mean_shoulder[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw the entire path of the mean positions for both head and shoulder
    for i in range(1, len(mean_head_positions)):
        cv.line(frame_copy, mean_head_positions[i - 1], mean_head_positions[i], (0, 255, 255), 2)
    for i in range(1, len(mean_shoulder_positions)):
        cv.line(frame_copy, mean_shoulder_positions[i - 1], mean_shoulder_positions[i], (255, 255, 0), 2)

    # Decide display based on video path: if "flat" is in the path, show bottom 50%; otherwise, show full frame
    if "flat" in video_path:
        display_frame = frame_copy[frame_copy.shape[0] // 2:, :]
    else:
        display_frame = frame_copy

    cv.imshow("Head and Shoulder Tracking", display_frame)

    # Initialize writer on first processed frame if not already created
    if writer is None:
        out_height, out_width = display_frame.shape[:2]
        # Create output folder based on the folder name in video_path
        folder_name = os.path.basename(os.path.dirname(video_path))
        output_folder = os.path.join("output/HeadShoulder", folder_name)
        os.makedirs(output_folder, exist_ok=True)
        # Create output file name (same as input base name with .mp4 extension)
        file_name = os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
        output_path = os.path.join(output_folder, file_name)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    writer.write(display_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

clip.release()
if writer is not None:
    writer.release()
cv.destroyAllWindows()
