import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from ultralytics import YOLO
import joblib
from pathlib import Path
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import datetime
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import json


# Load SimCLR model
def load_simclr_model(model_path, device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Load SVM models
def load_svm_models(svm_dir):
    svm_models = {}
    for svm_file in os.listdir(svm_dir):
        if svm_file.endswith('.joblib'):
            species = svm_file.split('_')[0]
            svm_models[species] = joblib.load(os.path.join(svm_dir, svm_file))
    return svm_models

# Extract features using SimCLR
def extract_features(simclr_model, img, device):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = simclr_model(img_t)
        #print(features)

    return features.squeeze(0).cpu().numpy()

def get_species_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        annotations = data['annotations']
    
    my_species =  {}
    for item in annotations:
        image_id = item['image_id']
        my_species[image_id+'.jpg'] = item['category_id']
    return my_species

# Run YOLO model and detect anomalies
def detect_anomalies(yolo_model, image_path, simclr_model, svm_models, device, anomaly_dir):
    str_image_path = str(image_path)
    img = Image.open(str_image_path)

    # Run YOLO prediction and save output with bounding boxes
    yolo_results = yolo_model.predict(source=str_image_path, verbose = False, save=False, imgsz=640, conf=0.25)

    features_list = []
    labels_list = []
    anomaly_count = 0

    # Check if results have boxes and process each detection
    if yolo_results and hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
    #if yolo_results[0].boxes is not None:
        boxes = yolo_results[0].boxes.xywh.cpu()  # Extract bounding boxes in XYWH format
   
        class_values = yolo_results[0].boxes.cls.cpu()

         # Save Anomalous Images
        bbox_dir = os.path.join(anomaly_dir, "bbox")
        os.makedirs(bbox_dir, exist_ok=True)

        
        idx = 0
        for box in boxes:
            x, y, w, h = box[:4].numpy()  # Convert tensor to numpy array
            #print(x, y, w, h)
            x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
            cropped_img = img.crop((x-w/2, y-h/2, x+w/2, y+h/2))
            cropped_img_filename = f"cropped_{idx}_{os.path.basename(str_image_path)}"
            cropped_img_path = os.path.join(bbox_dir, cropped_img_filename)
            # Save the cropped image
            #cropped_img.save(cropped_img_path)  #Turn on to confirm bounding boxes are cropped correctly
            features = extract_features(simclr_model, cropped_img, device)
            species_idx = class_values[idx] 
            species = get_species_name(species_idx)

            is_anomaly = species in svm_models and svm_models[species].predict([features])[0] == -1
            if is_anomaly:
                anomaly_count += 1

            features_list.append(features)
            labels_list.append(1 if is_anomaly else 0)
            idx = idx +1

    #print(f"Total features extracted: {len(features_list)}, Anomalies detected: {anomaly_count}")
    return features_list, labels_list


def get_species_name(cls_index): #ignore anomaly I thought I could make an anomaly class
    species_mapping = {
        0: 'anomaly',
        1: 'animal',
        2: 'bird',
        3: 'bobcat',
        4: 'boycot',
        5: 'c',
        6: 'coyote',
        7: 'dd',
        8: 'deer',
        9: 'dog',
        10: 'none',
        11: 'person',
        12: 'pig',
        13: 'pwe',
        14: 'raccoon',
        15: 'rodent',
        16: 'skunk',
    }
    return species_mapping.get(cls_index, 'unknown')

def get_transformed_features(features, species_labels, image_paths):
    filtered_features = [feat for feat in features if len(feat) > 0]
    filtered_species = [species_labels[i] for i, feat in enumerate(features) if len(feat) > 0]
    filtered_image_paths = [image_paths[i] for i, feat in enumerate(features) if len(feat) > 0]

    features_array = np.vstack(filtered_features)
    tsne = TSNE(n_components=2, random_state=0)
    transformed_features = tsne.fit_transform(features_array)

    return transformed_features, filtered_species, filtered_image_paths

#Calls DBScan for anomaly detection inside this function - bad programing, should be separated
def plot_tsne_clusters(transformed_features, filtered_species, species_color_map, output_dir):
    # Invert species_color_map
    inv_species_color_map = {v: "anomaly" if k == "anomaly" else get_species_name(k) for k, v in species_color_map.items()}

    # Map species to indices
    unique_species = list(set(filtered_species))
    species_to_index = {species: i for i, species in enumerate(unique_species)}

    # Assign colors and opacities based on indices
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_species)))

    # Species to opacity mapping
    species_opacity = {
        'person': 0.2,  # Example, replace with actual species and desired opacity
        #'skunk': 0.2,
        'animal': 0.2,
        'unknown': 0.6,
        'deer': 0.9,
        'coyote': 0.9,
        'bird': 0.9
        # Add more species and their opacities here
    }

        # Species to opacity mapping
    species_color = {
        'bobcat': 'red',  # Example, replace with actual species and desired opacity
        'anomaly': 'orange',
        'deer': 'mediumorchid',
        'skunk': 'magenta'
        # Add more species and their opacities here
    }

    # Plot with varying opacities
    for i, species in enumerate(unique_species):
        idx = [j for j, spec in enumerate(filtered_species) if spec == species]
        opacity = species_opacity.get(inv_species_color_map[species], 1.0)  # Default opacity is 1.0
        color = species_color.get(inv_species_color_map[species], colors[species_to_index[species]])
        plt.scatter(transformed_features[idx, 0], transformed_features[idx, 1], 
                    color=color, alpha=opacity, s=5)

    # Create a legend, only including species with opacity >= 0.5
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=inv_species_color_map[species],
                markersize=10, markerfacecolor=species_color.get(inv_species_color_map[species], colors[species_to_index[species]]),
                alpha=species_opacity.get(species, 1.0))
        for species in unique_species if species_opacity.get(inv_species_color_map[species], 1.0) >= 0.5
    ]
    plt.legend(handles=handles, title="Species")

    plt.title("t-SNE Visualization of Features")
    plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
    plt.close()

def detect_anomalies_with_dbscan(transformed_features, image_paths, output_dir, eps=2.5, min_samples=10):
    # Perform DBSCAN clustering

    #plot to evaluate eps
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(transformed_features)
    distances, indices = nn.kneighbors(transformed_features)

    # Sort the distances to the k-th nearest neighbor
    kth_distances = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(12, 6))
    plt.plot(kth_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {min_samples}-th nearest neighbor')
    plt.title('k-distance Plot')
    plt.grid(True)
    
    # Save the plot
    eps_path = os.path.join(output_dir, "dbscan_eps_evaluation.png")
    plt.savefig(eps_path)
    print(f"Plot saved to {eps_path}")
    plt.close()


    db = DBSCAN(eps=eps, min_samples=min_samples).fit(transformed_features)
    labels = db.labels_
    
    # Identify points classified as noise by DBSCAN (-1 label)
    anomalies_indices = np.where(labels == -1)[0]
    non_anomalies_indices = np.where(labels != -1)[0]
    
    # Print the number of anomalies detected
    print(f"Number of anomalies detected: {len(anomalies_indices)}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(transformed_features[non_anomalies_indices, 0], transformed_features[non_anomalies_indices, 1], 
                c='darkblue', label='Normal', s=30)
    plt.scatter(transformed_features[anomalies_indices, 0], transformed_features[anomalies_indices, 1], 
                c='darkorange', label='Anomaly', s=30)
    plt.title('DBSCAN Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "dbscan_anomaly_detection.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Save Anomalous Images
    anomaly_dir = os.path.join(output_dir, "anomalies")
    os.makedirs(anomaly_dir, exist_ok=True)
    for index in anomalies_indices:
        anomaly_image_path = image_paths[index]
        shutil.copy(anomaly_image_path, anomaly_dir)
        #print(f"Anomalous image saved to {anomaly_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique run ID
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    resnet_model_path = '/home/michael/animal/simclr_runs/300_epoch/encoder_epoch_250.pth'
    svm_models_dir = '/home/michael/animal/svm_models/'
    yolo_weights_path = '/home/michael/animal/runs/detect/train8/weights/best.pt'
    image_folder = Path('/home/michael/animal/unlabeled_anom_test/')
    json_path = Path('/home/michael/animal/jldp-animl-cct.json')

    simclr_model = load_simclr_model(resnet_model_path, device)
    svm_models = load_svm_models(svm_models_dir)
    yolo_model = YOLO(yolo_weights_path)
    output_dir = f"/home/michael/animal/anom_runs/run-{run_id}/tse_output"
    os.makedirs(output_dir, exist_ok=True)

    
    anomaly_dir = f"/home/michael/animal/anom_runs/run-{run_id}/images"
    os.makedirs(anomaly_dir, exist_ok=True)

    tsne_features = []
    tsne_labels = []
    tsne_image_paths = []

    species_color_map = {}  # Mapping of species to numerical labels
    current_label = 0

    species_map = get_species_from_json(json_path)



    for image_file in image_folder.glob('*.jpg'):
        file_name = image_file.name

        # Check if the filename begins with "anomaly"
        if file_name.startswith('anomaly'):
            species = 'anomaly'
        else:
            # Extract species name from the folder structure
            species = species_map.get(file_name, "Unknown")

        # Assign or retrieve numerical label for species
        if species not in species_color_map:
            species_color_map[species] = current_label
            current_label += 1
        numerical_label = species_color_map[species]

        features, _ = detect_anomalies(yolo_model, image_file, simclr_model, svm_models, device, anomaly_dir)
        for feature in features:
            tsne_features.append(feature)
            tsne_labels.append(numerical_label)
            tsne_image_paths.append(image_file)

   # print(species_color_map)

    # Generate and save t-SNE plot
    transformed_features, filtered_species, filtered_image_paths =  get_transformed_features(tsne_features, tsne_labels, tsne_image_paths)

    plot_tsne_clusters(transformed_features, filtered_species, species_color_map, output_dir)
    detect_anomalies_with_dbscan(transformed_features, filtered_image_paths, anomaly_dir, 2, 5)


if __name__ == '__main__':
    main()
