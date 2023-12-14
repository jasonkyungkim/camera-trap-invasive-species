import os
from PIL import Image
import random
import shutil

background_path = '/home/michael/animal/backgrounds'
overlay_path = '/home/michael/animal/anom_copy/3'
output_directory = '/home/michael/animal/anom_copy/18'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Assuming the error is in the crop_overlay function or similar
def crop_overlay(overlay_path, bbox):
    """
    Crops the overlay image based on bounding box coordinates.
    bbox - A tuple of (x, y, width, height) in relative terms of the image dimensions.
    """
    overlay_img = Image.open(overlay_path).convert("RGBA")
    img_width, img_height = overlay_img.size

    # Convert bbox coordinates to pixel values
    x = int(bbox[0] * img_width)
    y = int(bbox[1] * img_height)
    width = int(bbox[2] * img_width)
    height = int(bbox[3] * img_height)

    # Calculate the bounding box in terms of pixel values
    left = x - (width // 2)
    upper = y - (height // 2)
    right = x + (width // 2)
    lower = y + (height // 2)

    # Crop the overlay image to the bounding box size
    cropped_overlay = overlay_img.crop((left, upper, right, lower))
    return cropped_overlay


def overlay_image(background, overlay, output_path, bbox, min_size_percentage=0.1, max_size_percentage=0.5, transparency=180):
    background_img = Image.open(background)
    overlay_img = Image.open(overlay).convert("RGBA")

    # Calculate the actual pixel values for the bounding box based on the overlay image size
    img_width, img_height = overlay_img.size
    x_center = bbox[0] * img_width
    y_center = bbox[1] * img_height
    bbox_width = bbox[2] * img_width
    bbox_height = bbox[3] * img_height

    # Crop the overlay image based on the bounding box
    x_min = int(x_center - (bbox_width / 2))
    y_min = int(y_center - (bbox_height / 2))
    x_max = int(x_center + (bbox_width / 2))
    y_max = int(y_center + (bbox_height / 2))
    cropped_overlay = overlay_img.crop((x_min, y_min, x_max, y_max))

    # Determine the minimum and maximum size based on the background image size
    min_size = min(background_img.size) * min_size_percentage
    max_size = min(background_img.size) * max_size_percentage

    # Resize the cropped overlay to ensure it is not smaller than the minimum size
    if min(cropped_overlay.size) < min_size:
        scale_factor = min_size / min(cropped_overlay.size)
        new_width = int(cropped_overlay.width * scale_factor)
        new_height = int(cropped_overlay.height * scale_factor)
        cropped_overlay = cropped_overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Randomly scale the cropped overlay between the minimum and maximum size
    scale_factor = random.uniform(min_size_percentage, max_size_percentage)
    new_width = int(background_img.width * scale_factor)
    new_height = int(new_width / (cropped_overlay.width / cropped_overlay.height))
    resized_overlay = cropped_overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Make the resized overlay semi-transparent
    resized_overlay.putalpha(transparency)

    # Calculate max x and y positions for the overlay
    max_x = background_img.width - new_width
    max_y = background_img.height - new_height

    # Choose a random position to place the overlay
    position_x = random.randint(0, max_x) if max_x > 0 else 0
    position_y = random.randint(0, max_y) if max_y > 0 else 0

    # Paste the resized, semi-transparent overlay onto the background
    background_img.paste(resized_overlay, (position_x, position_y), resized_overlay)

    # Save the new image
    background_img.save(output_path)




def read_bounding_boxes(bbox_directory, overlay_filename_base):
    bbox_file = overlay_filename_base.replace('.JPG', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.JPEG', '.txt')
    bbox_path = os.path.join(bbox_directory, bbox_file)
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            bbox_data = f.readline().strip().split(' ')
            if len(bbox_data) >= 5: # Check for species_id followed by four float values
                species_id = bbox_data[0]
                center_x_rel, center_y_rel, width_rel, height_rel = map(float, bbox_data[1:])
                return (center_x_rel, center_y_rel, width_rel, height_rel) # Returns a tuple for bbox
            else:
                print(f"Unexpected format in {bbox_file}")
                return None # Return None to indicate invalid or non-existent bbox data
    else:
        print(f"Bbox file not found: {bbox_file}")
        return None



# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get lists of files
background_files = [f for f in os.listdir(background_path) if f.endswith('.jpg')]
overlay_files = [f for f in os.listdir(overlay_path) if f.endswith('.jpg')]

# Shuffle background images to ensure randomness
random.shuffle(background_files)

# Limit each overlay image to be used on at most three different backgrounds
for overlay_file in overlay_files:
    chosen_backgrounds = random.sample(background_files, k=min(3, len(background_files)))
    for background_file in chosen_backgrounds:
        background_img_path = os.path.join(background_path, background_file)
        overlay_img_path = os.path.join(overlay_path, overlay_file)
        output_img_path = os.path.join(output_directory, f'anomaly_{background_file[:-4]}_{overlay_file}')

        # Read the bounding box for the overlay image
        bbox_info = read_bounding_boxes(overlay_path, overlay_file)
        if bbox_info:
            # Unpack the bounding box information, without expecting species_id
            center_x_rel, center_y_rel, width_rel, height_rel = bbox_info

            # Create and save the synthetic anomaly image
            overlay_image(background_img_path, overlay_img_path, output_img_path, bbox_info)


        # Copy the corresponding bbox file to the output directory
        bbox_text_file = overlay_file.replace('.JPG', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.JPEG', '.txt')
        shutil.copy(os.path.join(overlay_path, bbox_text_file), output_directory)  # Copy bbox file to output dir

print("Synthetic anomalies created successfully!")