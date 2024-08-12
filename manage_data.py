import os
import shutil

# Define the source and destination directories
source_dir = "/home/habibi/Uni/SS24/DLAM"
destination_dir = os.path.join(source_dir, "dataa")

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through all files in the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".mat"):
            # Construct full file paths
            file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_dir, file)

            # If the file exists, append a number to the filename
            base_name, extension = os.path.splitext(file)
            counter = 1
            new_file_path = destination_file_path

            while os.path.exists(new_file_path):
                new_file_name = f"{base_name}_{counter}{extension}"
                new_file_path = os.path.join(destination_dir, new_file_name)
                counter += 1

            # Move the file to the (potentially renamed) destination
            shutil.move(file_path, new_file_path)

print("All .mat files have been moved to the 'dataa' folder.")
