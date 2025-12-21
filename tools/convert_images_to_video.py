import cv2
import os
import argparse
import glob
from tqdm import tqdm

def convert_images_to_video(input_folder, output_file, fps=25):
    # Search for images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Sort images to ensure correct order
    images.sort()
    
    if not images:
        print(f"Error: No images found in {input_folder}")
        return

    print(f"Found {len(images)} images in {input_folder}")
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f"Converting to {output_file} ({width}x{height} @ {fps}fps)...")
    
    for filename in tqdm(images):
        img = cv2.imread(filename)
        out.write(img)
        
    out.release()
    print("Conversion finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image sequence to a video file.")
    parser.add_argument("input_folder", help="Path to the folder containing images")
    parser.add_argument("output_file", help="Path for the output video (e.g., data/video.mp4)")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate (default: 25)")
    #python tools/convert_images_to_video.py /path/to/your/images /path/to/your/video.mp4 --fps 30

    args = parser.parse_args()
    
    convert_images_to_video(args.input_folder, args.output_file, args.fps)
