import cv2 as cv

# Set start and end seconds (adjust these as needed)
start_sec = 15   # For example, start at 10 seconds
end_sec = 30     # For example, end at 20 seconds

# Open the video file
clip = cv.VideoCapture("/cs/usr/ronhafzadi/Downloads/v4.mp4")

# Check if the video was opened successfully
if not clip.isOpened():
    print("Error: Video file not found or cannot be opened.")
    exit()

# Set the starting position (in milliseconds)
clip.set(cv.CAP_PROP_POS_MSEC, start_sec * 1000)

# Retrieve video properties
fps = clip.get(cv.CAP_PROP_FPS)
width = int(clip.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(clip.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

# Create a VideoWriter object to save the output clip as "clip1.mp4"
out = cv.VideoWriter("input/hillp/r1.mp4", fourcc, fps, (width, height))

while True:
    # Get the current time position in seconds
    current_time = clip.get(cv.CAP_PROP_POS_MSEC) / 1000
    if current_time > end_sec:
        break

    ret, frame = clip.read()
    if not ret:
        break

    # Write the frame to the output file without any rotation
    out.write(frame)

    # Optionally display the frame (press 'q' to quit early)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
clip.release()
out.release()
cv.destroyAllWindows()
