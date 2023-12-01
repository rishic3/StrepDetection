import os
import shutil
import random

def sample_images_and_cams(src_directory, dest_directory, sample_size=4):
    subdirectories = ["False_Negatives", "True_Negatives", "True_Positives", "False_Positives"]

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    for subdir in subdirectories:
        subdir_path = os.path.join(src_directory, subdir)
        dest_subdir_path = os.path.join(dest_directory, subdir)

        if not os.path.exists(dest_subdir_path):
            os.makedirs(dest_subdir_path)

        # List and shuffle image files
        image_files = [f for f in os.listdir(subdir_path) if f.startswith("image_")]
        random.shuffle(image_files)

        # Sample images
        sampled_images = image_files[:sample_size]

        for image in sampled_images:
            # Copy image file
            src_image_path = os.path.join(subdir_path, image)
            dest_image_path = os.path.join(dest_subdir_path, image)
            shutil.copy(src_image_path, dest_image_path)

            # Copy corresponding CAM file
            cam_file = image.replace("image_", "cam_")
            src_cam_path = os.path.join(subdir_path, cam_file)
            dest_cam_path = os.path.join(dest_subdir_path, cam_file)
            shutil.copy(src_cam_path, dest_cam_path)

            print(f"Copied {src_image_path} and {src_cam_path} to {dest_subdir_path}")

# Specify the source and destination directories
src_directory = "/data/datasets/rishi/symptom_classification/output_images"  # Replace with the path to your 'output_images' directory
dest_directory = "/data/datasets/rishi/symptom_classification/output_image_samples"  # Replace with your desired destination directory path
sample_images_and_cams(src_directory, dest_directory)
