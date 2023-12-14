import os
import json
import shutil
from PIL import Image

# Load the JSON file
with open("jldp-animl-cct.json", 'r') as f:
    data = json.load(f)

# Create the main output directories if they don't exist
os.makedirs("processed_v2", exist_ok=True)
os.makedirs("no_bbox_v2", exist_ok=True)

def find_image(image_id, root_folder):
    adjusted_image_id = image_id.replace("jldp:", "")
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if image_id in file or adjusted_image_id in file:
                image_path = os.path.join(root, file)
                #print("Found image path:", image_path)  # Debug print
                return image_path
    print(f"Image not found for ID: {image_id}")  # Debug print
    return None



missing_images_file = open("missing_images.txt", "w")



# Create a dictionary to map image IDs to image details
image_details = {item['id']: item for item in data['images']}

# Process each bounding box annotation
for item in data['annotations']:
    image_id = item['image_id']
    image_path = find_image(image_id, 'jldp-animl-images')

    if not image_path:
        missing_images_file.write(f"{image_id}\n")
        continue

    # Extract width and height either from the JSON or from the actual image
    img_width = img_height = None
    if 'width' in image_details[image_id] and 'height' in image_details[image_id]:
        img_width = image_details[image_id]['width']
        img_height = image_details[image_id]['height']
    else:
        with Image.open(image_path) as img:
            img_width, img_height = img.size

    if all(val is not None for val in item.get('bbox', [])):
        # Ensure the category folder exists inside 'processed'
        category_folder = os.path.join("processed_v2", str(item['category_id']))
        os.makedirs(category_folder, exist_ok=True)

        with Image.open(image_path) as img:
            img = img.resize((640, 640))  # Resize for YOLO input
            img.save(os.path.join(category_folder, f"{image_id}.jpg"))

        with open(os.path.join(category_folder, f"{image_id}.txt"), 'w') as f:
            x, y, width, height = item['bbox']
            x_center, y_center = (x + width / 2) / img_width, (y + height / 2) / img_height
            width, height = width / img_width, height / img_height
            f.write(f"{item['category_id']} {x_center} {y_center} {width} {height}\n")
    else:
        # If category_id is available, group by category_id inside 'no_bbox', otherwise just put in 'no_bbox'
        no_bbox_category_folder = os.path.join("no_bbox_v2", str(item.get('category_id', '')))
        os.makedirs(no_bbox_category_folder, exist_ok=True)
        shutil.copy(image_path, os.path.join(no_bbox_category_folder, f"{image_id}.jpg"))

missing_images_file.close()

# Define source and destination directories
source_dir = 'no_bbox_v2'
destination_dir = 'images_for_detection_v2'
os.makedirs(destination_dir, exist_ok=True)

# Loop through each category_id folder in no_bbox and copy images to the destination
for category_folder in os.listdir(source_dir):
    category_folder_path = os.path.join(source_dir, category_folder)
    
    # Ensure it's a directory before proceeding
    if os.path.isdir(category_folder_path):
        for image_file in os.listdir(category_folder_path):
            source_path = os.path.join(category_folder_path, image_file)
            destination_path = os.path.join(destination_dir, image_file)
            
            # Copy the image to the destination directory
            shutil.copy(source_path, destination_path)

print("Images copied successfully!")
