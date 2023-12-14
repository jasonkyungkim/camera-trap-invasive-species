import os
import shutil
import random

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def setup_yolo_directory_structure(base_dir):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir

def copy_files(src_files, images_dir, labels_dir):
    for file in src_files:
        if file.lower().endswith('.jpg'):
            shutil.copy(file, images_dir)
            # Copy the corresponding .txt file if it exists
            txt_file = os.path.splitext(file)[0] + '.txt'
            if os.path.exists(txt_file):
                shutil.copy(txt_file, labels_dir)

def create_train_val_split(source_dir, train_dir, val_dir, train_ratio=0.8):
    clear_directory(train_dir)
    clear_directory(val_dir)

    train_images_dir, train_labels_dir = setup_yolo_directory_structure(train_dir)
    val_images_dir, val_labels_dir = setup_yolo_directory_structure(val_dir)

    for species in os.listdir(source_dir):
        species_dir = os.path.join(source_dir, species)
        if not os.path.isdir(species_dir):
            continue

        images = [os.path.join(species_dir, f) for f in os.listdir(species_dir) if f.lower().endswith('.jpg')]
        random.shuffle(images)

        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        copy_files(train_images, train_images_dir, train_labels_dir)
        copy_files(val_images, val_images_dir, val_labels_dir)

# Paths
source_dir = '/home/michael/animal/anom_training/'
train_dir = '/home/michael/animal/yolo_train'
val_dir = '/home/michael/animal/yolo_val'

# Create the train-val split
create_train_val_split(source_dir, train_dir, val_dir)
