import os
import torch
from sklearn.svm import OneClassSVM
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import joblib
import json


def adjust_bbox_for_resized_image(original_bbox, original_width, original_height):
    resized_width, resized_height = 640, 640
    x_min, y_min, width, height = original_bbox
    x_min_scaled = x_min * (resized_width / original_width)
    y_min_scaled = y_min * (resized_height / original_height)
    width_scaled = width * (resized_width / original_width)
    height_scaled = height * (resized_height / original_height)
    return [x_min_scaled, y_min_scaled, width_scaled, height_scaled]

def read_bounding_box_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        annotations = data['annotations']
        images_info = {img['id']: img for img in data['images']}

    bbox_data = {}
    for item in annotations:
        image_id = item['image_id']
        original_img_info = images_info.get(image_id, {})
        original_width = original_img_info.get('width')
        original_height = original_img_info.get('height')

        if all([original_width, original_height]):
            bbox = item.get('bbox', [])
            if all(val is not None for val in bbox) and bbox[2] > 0 and bbox[3] > 0:
                adjusted_bbox = adjust_bbox_for_resized_image(bbox, original_width, original_height)
                bbox_data[image_id] = adjusted_bbox
    return bbox_data

def crop_resize_image(image, bbox):
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize((224, 224))
    return cropped_image

def load_simclr_model(model_path, device):
    # Load the base ResNet model
    model = resnet50(weights=None)  # Use 'weights=None' to initialize without pretrained weights
    model.fc = torch.nn.Identity()  # Replace the fully connected layer as per SimCLR architecture

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Function to extract features using SimCLR
def extract_features(model, image_path, bbox=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with Image.open(image_path).convert('RGB') as img:
        if bbox:
            img = crop_resize_image(img, bbox)  # Crop image to the bounding box
        img_t = transform(img).unsqueeze(0)  # Add batch dimension

    img_t = img_t.to(next(model.parameters()).device)

    with torch.no_grad():
        features = model(img_t)

    return features.squeeze(0).cpu().numpy()

def train_one_class_svm_for_species(train_dir, simclr_model, bbox_data):
    svm_models = {}

    for species_folder in os.listdir(train_dir):
        species_path = os.path.join(train_dir, species_folder)
        if os.path.isdir(species_path):
            features = []
            for image_file in os.listdir(species_path):
                image_path = os.path.join(species_path, image_file)
                image_id = Path(image_path).stem
                bbox = bbox_data.get(image_id)

                image_features = extract_features(simclr_model, image_path, bbox)
                if image_features is not None:
                    features.append(image_features)

            if features:
                svm = OneClassSVM(gamma='auto').fit(features)
                svm_models[species_folder] = svm
            else:
                print(f"No features extracted for species: {species_folder}")

    return svm_models


def save_svm_models(svm_models, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through each species and its corresponding SVM model
    for species, svm_model in svm_models.items():
        # Define the path to save each model
        save_path = os.path.join(save_dir, f'{species}_svm_model.joblib')

        # Save the SVM model
        joblib.dump(svm_model, save_path)
        print(f"Saved SVM model for {species} at {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    simclr_model_path = '/home/michael/animal/simclr_runs/300_epoch/encoder_epoch_250.pth' 
    simclr_model = load_simclr_model(simclr_model_path, device)
    
    #Load bounding box data
    json_path = Path('/home/michael/animal/jldp-animl-cct.json')
    bbox_data = read_bounding_box_data(json_path)

    train_dir = '/home/michael/animal/sm_train/'
    save_dir = '/home/michael/animal/svm_models/'
    

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)


    # Train the SVM models for each species
    svm_models = train_one_class_svm_for_species(train_dir, simclr_model, bbox_data)

    # Save the trained SVM models
    save_svm_models(svm_models, save_dir)

if __name__ == '__main__':
    main()