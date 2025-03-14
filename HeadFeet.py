import cv2 as cv
import numpy as np
from collections import deque
import os

AVG = 30
video_path = "input/incline/r1.mp4"
clip = cv.VideoCapture(video_path)

bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
kernel = np.ones((3, 3), np.uint8)

# To track direction and distances
prev_centroid_x = None
walking_direction = None
distance_history = deque(maxlen=AVG)


# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Function to determine terrain slope based on last 30 frames
def determine_slope(last_30_frames):
    if len(last_30_frames) < AVG:
        return "flat"
    upward = 0
    downward = 0
    for entry in last_30_frames:
        dist_leading = entry['dist_leading']
        dist_other = entry['dist_other']
        if dist_leading < dist_other:
            upward += 1
        else:
            downward += 1
    return "upward" if upward > downward else "downward"


writer = None
fps = clip.get(cv.CAP_PROP_FPS)

while True:
    ret, frame = clip.read()
    if not ret:
        break

    # Process frame as usual
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fgmask = bg_subtractor.apply(gray_frame)
    _, bright_mask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
    cleaned_mask = cv.erode(bright_mask, kernel, iterations=1)
    cleaned_mask = cv.dilate(cleaned_mask, kernel, iterations=2)
    contours, _ = cv.findContours(cleaned_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    frame_copy = frame.copy()
    foot_candidates = []
    head_position = None
    current_centroid_x = None

    for c in contours:
        if cv.contourArea(c) > 1000:  # Adjust as needed
            x, y, width, height = cv.boundingRect(c)
            current_centroid_x = x + width // 2

            # HEAD DETECTION using Hough Circle (Top 15% of the bounding box)
            top_region = frame[y: y + int(height * 0.15), x:x + width]
            top_gray = cv.cvtColor(top_region, cv.COLOR_BGR2GRAY)
            circles = cv.HoughCircles(top_gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                      param1=50, param2=30, minRadius=5, maxRadius=50)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    head_position = (x + cx, y + cy)
                    cv.circle(frame_copy, (x + cx, y + cy), r, (0, 0, 255), 4)

            # FOOT DETECTION (Bottom 30% of bounding box)
            lower_region = frame[y + int(height * 0.8): y + height, x:x + width]
            lower_gray = cv.cvtColor(lower_region, cv.COLOR_BGR2GRAY)
            _, shoes_mask = cv.threshold(lower_gray, 160, 255, cv.THRESH_BINARY)
            shoe_contours, _ = cv.findContours(shoes_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for shoe in shoe_contours:
                if cv.contourArea(shoe) > 30:  # Filter small white areas
                    sx, sy, sw, sh = cv.boundingRect(shoe)
                    cv.rectangle(frame_copy, (x + sx, y + int(height * 0.8) + sy),
                                 (x + sx + sw, y + int(height * 0.8) + sy + sh), (0, 255, 0), 2)
                    foot_candidates.append((x + sx + sw // 2, y + int(height * 0.8) + sy + sh))

    # Determine Walking Direction
    if prev_centroid_x is not None and current_centroid_x is not None:
        if current_centroid_x - prev_centroid_x > 5:
            walking_direction = "right"
        elif current_centroid_x - prev_centroid_x < -5:
            walking_direction = "left"
    prev_centroid_x = current_centroid_x

    # Identify Leading Foot and Compute Distances
    if len(foot_candidates) >= 2 and head_position:
        foot_candidates = sorted(foot_candidates, key=lambda f: f[0])
        if walking_direction == "right":
            leading = foot_candidates[-1]
            other = foot_candidates[0]
        elif walking_direction == "left":
            leading = foot_candidates[0]
            other = foot_candidates[-1]
        else:
            leading = foot_candidates[0]
            other = foot_candidates[1]
        dist_leading = calculate_distance(head_position, leading)
        dist_other = calculate_distance(head_position, other)
        distance_history.append({'dist_leading': dist_leading, 'dist_other': dist_other})
        cv.line(frame_copy, head_position, leading, (255, 0, 0), 2)
        cv.line(frame_copy, head_position, other, (0, 255, 255), 2)
        cv.putText(frame_copy, f"Dist(lead): {dist_leading:.1f}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv.putText(frame_copy, f"Dist(other): {dist_other:.1f}", (20, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Determine Terrain Slope
    slope = determine_slope(distance_history)
    cv.putText(frame_copy, f"Terrain Slope: {slope.upper()}", (20, 100),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the full processed frame
    display_frame = frame_copy
    cv.imshow("Detected Shoes and Terrain Analysis", display_frame)

    # Saving Option: Save the output video
    if writer is None:
        out_height, out_width = display_frame.shape[:2]
        folder_name = os.path.basename(os.path.dirname(video_path))
        file_base = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join("output/HeadFeet", folder_name)
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
