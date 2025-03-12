import cv2
import os
import glob

# Define the output filename
output_video = "winners.mp4"

# Get all matching video files
video_files = sorted(glob.glob("evol10_structure_gen*_iter*.mp4"))

if not video_files:
    print("No matching videos found!")
    exit()

# Read the first video to get the frame size
first_video = cv2.VideoCapture(video_files[0])
frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(first_video.get(cv2.CAP_PROP_FPS))
first_video.release()

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Process each video
for video_file in video_files:
    print(f"Adding {video_file} to combined video...")
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Resize frame if necessary (optional, adjust if dimensions vary)
        frame = cv2.resize(frame, (frame_width, frame_height))
        out.write(frame)

    cap.release()

# Release the writer
out.release()
print(f"Combined video saved as {output_video}")
