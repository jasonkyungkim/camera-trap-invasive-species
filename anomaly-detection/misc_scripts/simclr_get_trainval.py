import os
import shutil
import random

def create_train_val_split(source_dir, train_dir, val_dir, val_split=0.2):
    """
    Creates training and validation datasets from the source directory.
    
    :param source_dir: Path to the source directory containing class subfolders
    :param train_dir: Path to the training directory
    :param val_dir: Path to the validation directory
    :param val_split: Fraction of images to be used for validation
    """
    # Clear existing data in train and val directories
    for directory in [train_dir, val_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    # Iterate through each class/subfolder in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue

        # Create corresponding class directories in train and val folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all image files in the class directory
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        # Split images into training and validation sets
        val_count = int(len(images) * val_split)
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Copy images to the respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

# Define your paths
source_dir = '/home/michael/animal/anom_training/'
train_dir = '/home/michael/animal/sm_train/'
val_dir = '/home/michael/animal/sm_val/'

# Create the datasets
create_train_val_split(source_dir, train_dir, val_dir, val_split=0.2)
