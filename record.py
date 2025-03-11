import cv2
import os

def create_video_from_images(image_folder, output_video, fps=20):
    # Get sorted list of image file names in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    
    if not images:
        print("No images found in", image_folder)
        return

    # Read the first image to determine the video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print("Video saved as", output_video)

if __name__ == '__main__':
    # For example, record a video from images in the 'diffmpm/iter001/' folder.
    create_video_from_images('diffmpm/iter004', 'structure_video.mp4')
