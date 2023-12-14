import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
import os
import random
import datetime
from pathlib import Path
import json
from torch.utils.data._utils.collate import default_collate

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


torch.cuda.empty_cache()


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



class BaseImageDataset(Dataset):
    def __init__(self, root_dir, bbox_data, transform=None, save_sample_images=False, save_dir='sample_images'):
        self.root_dir = root_dir
        self.bbox_data = bbox_data
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.save_sample_images = save_sample_images
        self.save_dir = save_dir
        self.image_counter = 0
        if self.save_sample_images and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_transform_image(self, image_path):
        try:
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                raise ValueError(f"Image not found or is not a file: {image_path}")

            with Image.open(image_path) as img:
                if img.width == 0 or img.height == 0:
                    raise ValueError(f"Invalid image dimensions: {image_path}")

                img = img.convert('RGB')
                image_id = Path(image_path).stem
                bbox = self.bbox_data.get(image_id, None)

                if bbox:
                    img = self.crop_resize_image(img, bbox)

            if self.save_sample_images and self.image_counter % 100 == 0:
                self.save_transformed_image(img, image_id)
            self.image_counter += 1

            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros([3, 224, 224])

    def crop_resize_image(self, image, bbox):
        x_min, y_min, width, height = bbox
        x_max, y_max = x_min + width, y_min + height
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image = cropped_image.resize((224, 224))
        return cropped_image

    def save_transformed_image(self, img, image_id):
        save_path = os.path.join(self.save_dir, f"{image_id}_transformed.jpg")
        img.save(save_path)


class TripletDataset(BaseImageDataset):
    def __init__(self, root_dir, bbox_data, transform=None, save_sample_images=False, save_dir='sample_images'):
        super().__init__(root_dir, bbox_data, transform, save_sample_images, save_dir)
        self.species_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_tuples = self._create_image_tuples()

    def _create_image_tuples(self):
        image_tuples = []
        for species_folder in self.species_folders:
            species_path = os.path.join(self.root_dir, species_folder)
            images = [img for img in os.listdir(species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for anchor_image in images:
                positive_images = [img for img in images if img != anchor_image]
                if positive_images:
                    negative_species_folder = random.choice([d for d in self.species_folders if d != species_folder])
                    negative_species_path = os.path.join(self.root_dir, negative_species_folder)
                    negative_images = [img for img in os.listdir(negative_species_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if negative_images:
                        negative_image = random.choice(negative_images)
                        image_tuples.append((os.path.join(species_path, anchor_image), 
                                             os.path.join(species_path, random.choice(positive_images)), 
                                             os.path.join(negative_species_path, negative_image)))
        return image_tuples

    def __len__(self):
        return len(self.image_tuples)

    def __getitem__(self, index):
        anchor_path, positive_path, negative_path = self.image_tuples[index]
        anchor = self.load_transform_image(anchor_path)
        positive = self.load_transform_image(positive_path)
        negative = self.load_transform_image(negative_path)
        return anchor, positive, negative

class SimCLRDataset(BaseImageDataset):
    def __init__(self, root_dir, bbox_data, transform=None, save_sample_images=False, save_dir='sample_images'):
        super().__init__(root_dir, bbox_data, transform, save_sample_images, save_dir)
        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self):
        image_paths = []
        for species_folder in os.listdir(self.root_dir):
            species_path = os.path.join(self.root_dir, species_folder)
            if os.path.isdir(species_path):
                for image_name in os.listdir(species_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(species_path, image_name))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_i, image_j = self.load_transform_images(image_path)
        return image_i, image_j

#######################################

 

def train_and_validate(encoder, train_loader, val_loader, epochs, optimizer, criterion, device, run_folder):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Training
        total_train_loss = 0
        encoder.train()
        for batch in train_loader:
            batch = tuple(filter(lambda x: x is not None, batch))
            if not batch or any(item is None for triplet in batch for item in triplet):
                continue  # Skip batch if it's empty or contains None
            anchor, positive, negative = batch
            anchor = torch.cat(anchor, dim=0).to(device)
            positive = torch.cat(positive, dim=0).to(device)
            negative = torch.cat(negative, dim=0).to(device)

            # Extract features and compute loss
            anchor_out = encoder(anchor)
            positive_out = encoder(positive)
            negative_out = encoder(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        total_val_loss = 0
        encoder.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(filter(lambda x: x is not None, batch))
                if not batch or any(item is None for triplet in batch for item in triplet):
                    continue  # Skip batch if it's empty or contains None
                anchor, positive, negative = batch
                anchor = torch.cat(anchor, dim=0).to(device)
                positive = torch.cat(positive, dim=0).to(device)
                negative = torch.cat(negative, dim=0).to(device)

                # Extract features and compute loss
                anchor_out = encoder(anchor)
                positive_out = encoder(positive)
                negative_out = encoder(negative)

                loss = criterion(anchor_out, positive_out, negative_out)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)

        # Save the model state
        encoder_save_path = os.path.join(run_folder, f"encoder_epoch_{epoch}.pth")
        torch.save(encoder.state_dict(), encoder_save_path)

        print(f"Epoch {epoch + 1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_folder, 'loss_plot.png'))




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = '/home/michael/animal/sm_train/'
    val_dir = '/home/michael/animal/sm_val/'

    #Load bounding box data
    json_path = Path('/home/michael/animal/jldp-animl-cct.json')
    bbox_data = read_bounding_box_data(json_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = f"simclr_runs/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    save_dir = os.path.join(run_folder, 'sample_images')
    print(f"Sample images will be saved in: {save_dir}") 

    #Transform SimCLR
    transform = TransformsSimCLR(size=224) #probably too small but my computer was running out of memory even on a powerful computer

    # Initialize the datasets
    train_dataset = TripletDataset(root_dir=train_dir, bbox_data=bbox_data, transform=transform, save_sample_images=True, save_dir=save_dir)
    val_dataset = TripletDataset(root_dir=val_dir, bbox_data=bbox_data, transform=transform, save_sample_images=True, save_dir=save_dir)

    # Initialize the SimCLR dataset with bounding box data
    simclr_train_dataset = SimCLRDataset(root_dir=train_dir, bbox_data=bbox_data, transform=transform)
    simclr_val_dataset = SimCLRDataset(root_dir=val_dir, bbox_data=bbox_data, transform=transform)

    
    class_counts = defaultdict(int)
    for species_folder in os.listdir(train_dir):
        species_path = os.path.join(train_dir, species_folder)
        for image_name in os.listdir(species_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_counts[species_folder] += 1

    weights = []
    for species_folder in os.listdir(train_dir):
        species_path = os.path.join(train_dir, species_folder)
        for image_name in os.listdir(species_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                weights.append(1.0 / class_counts[species_folder])

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def safe_collate(batch):
        batch = [item for item in batch if item is not None and not any(x is None for x in item)]
        if not batch:
            # Return zero tensors for an entire empty batch
            zero_tensor = torch.zeros([3, 224, 224])
            return zero_tensor, zero_tensor, zero_tensor
        return default_collate(batch)



    # Training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=sampler, num_workers=4, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=safe_collate)



    
    # Initialize the model
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    n_features = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)

    # Initialize SimCLR with the encoder
    projection_dim = 64
    model = SimCLR(encoder, projection_dim, n_features).to(device)

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = TripletMarginLoss(margin=1.0)

    

    # Train and validate the model
    train_and_validate(encoder, train_loader, val_loader, epochs=300, optimizer=optimizer, criterion=criterion, device=device, run_folder=run_folder)


if __name__ == '__main__':
    main()

