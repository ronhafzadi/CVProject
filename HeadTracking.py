import cv2 as cv
import numpy as np
from collections import deque
import os

AVG = 30
video_path = "input/flat/l1.mp4"  # Change this to your video path
clip = cv.VideoCapture(video_path)

bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)
kernel = np.ones((5, 5), np.uint8)

# Store head positions for path drawing and mean calculation
head_path = deque(maxlen=100)  # For drawing the complete head path
head_positions = deque(maxlen=20)  # For calculating the mean head position
mean_path = deque(maxlen=20)  # For storing mean head positions

# Setup for saving video output
writer = None
fps = clip.get(cv.CAP_PROP_FPS)

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

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Process only the largest contour (assuming it's the person)
    if contours and cv.contourArea(contours[0]) > 1000:
        contour = contours[0]
        x, y, w, h = cv.boundingRect(contour)

        # Draw bounding box around the person
        cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define head region as the top 20% of the bounding box
        head_region_y = y
        head_region_h = int(h * 0.2)
        head_region = thresh[head_region_y:head_region_y + head_region_h, x:x + w]

        # Draw the head region rectangle
        cv.rectangle(frame_copy, (x, head_region_y), (x + w, head_region_y + head_region_h), (255, 0, 0), 2)

        if head_region.size > 0:
            # Find contours in the head region
            head_contours, _ = cv.findContours(head_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if head_contours:
                # Use the largest blob in the head region
                head_contour = max(head_contours, key=cv.contourArea)
                if cv.contourArea(head_contour) > 50:
                    M = cv.moments(head_contour)
                    if M["m00"] != 0:
                        head_x = int(M["m10"] / M["m00"]) + x
                        head_y = int(M["m01"] / M["m00"]) + head_region_y
                        head_position = (head_x, head_y)
                        head_found = True

            # If no good contour is found, use the center of the upper part
            if not head_found:
                head_position = (x + w // 2, y + head_region_h // 2)
                head_found = True

            # Add head position and draw it
            if head_found:
                head_path.append(head_position)
                head_positions.append(head_position)
                cv.circle(frame_copy, head_position, 10, (0, 0, 255), 2)
                cv.putText(frame_copy, "Head", (head_position[0] + 10, head_position[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the head path
    for i in range(1, len(head_path)):
        if head_path[i - 1] and head_path[i]:
            cv.line(frame_copy, head_path[i - 1], head_path[i], (255, 0, 0), 2)

    # Calculate mean head position
    if len(head_positions) > 5:
        xs = [p[0] for p in head_positions]
        ys = [p[1] for p in head_positions]
        mean_head = (int(np.mean(xs)), int(np.mean(ys)))
        mean_path.append(mean_head)
        cv.circle(frame_copy, mean_head, 5, (0, 255, 255), -1)
        cv.putText(frame_copy, "Mean", (mean_head[0] + 10, mean_head[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw mean path
    for i in range(1, len(mean_path)):
        cv.line(frame_copy, mean_path[i - 1], mean_path[i], (0, 255, 0), 2)

    # Calculate angle of the path and determine slope (terrain analysis)
    if len(mean_path) >= 5:
        first_point = mean_path[0]
        last_point = mean_path[-1]
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if angle < 0:
            angle += 360
        moving_right_to_left = dx < 0
        slope_angle = abs(angle)
        if slope_angle > 90:
            slope_angle = 180 - slope_angle
        if slope_angle > 90:
            slope_angle = 180 - slope_angle
        is_acute = (0 <= angle <= 90) or (270 <= angle <= 360)
        slope_direction = ""
        if moving_right_to_left:
            if is_acute:
                slope_direction = "Downwards Slope" if angle <= 180 else "Upwards Slope"
            else:
                slope_direction = "Upwards Slope" if angle > 180 else "Downwards Slope"
        else:
            if is_acute:
                slope_direction = "Upwards Slope" if angle > 180 else "Downwards Slope"
            else:
                slope_direction = "Downwards Slope" if angle <= 180 else "Upwards Slope"

        cv.putText(frame_copy, f"Angle: {angle:.1f} deg", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(frame_copy, f"Direction: {'Right to Left' if moving_right_to_left else 'Left to Right'}",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(frame_copy, f"Slope: {slope_direction}",
                   (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        line_length = 100
        end_x = int(last_point[0] + line_length * np.cos(np.radians(angle)))
        end_y = int(last_point[1] + line_length * np.sin(np.radians(angle)))
        cv.line(frame_copy, last_point, (end_x, end_y), (0, 255, 255), 2)

    # Decide display based on video path: if "flat" is in the path, show bottom 50%; otherwise, show full frame
    display_frame = frame_copy

    cv.imshow("Head Tracking", display_frame)

    # Initialize VideoWriter on first frame if not already done
    if writer is None:
        out_height, out_width = display_frame.shape[:2]
        # Extract folder name and file base from video_path
        folder_name = os.path.basename(os.path.dirname(video_path))
        file_base = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join("output/HeadTracking", folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_base + ".mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    writer.write(display_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

clip.release()
if writer is not None:
    writer.release()
cv.destroyAllWindows()
