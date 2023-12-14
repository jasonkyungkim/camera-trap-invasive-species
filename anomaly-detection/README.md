# Anomaly Detection In Wildlife Images

_Note: These scripts assumes that you have already pre-processed your images according to YOLO's requirements, and have already created a train and validation folder. It is also very specific to the animl dataset - it works directly with the json folder, etc.

## Requirements and Dependencies

- Python 3.8+
- PyTorch 1.7+ : deep learning framework
- torchvision 0.8+ : used for pre-trained models and common image transformations
- Pil(Pillow) : image handling
- NumPy
- MatplotLib
- sklearn
- joblib: trading a loading saved models
- json 
- os, pathlib, shutil: file and directory handling
- ultranalytics: YOLO: Object detection model
- datetime

## Adjustable Filepaths 

### SimCLR Script

- root_dir 
- train_dir
- val_dir
- json_path 
- save_dir
- simclr


### OneClassSVM Script

- sim_clr_model_path: adjust to load different SimClr weights
- json_path 
- train_dir
- save_dir : where the OneClassSVM models will be saved

### Anomaly Detect Script

- resnet_model_path: path to SimCLR model (not different than above)
- svm_models_dir: Directory containing OneClassSVM models
- yolo_weights_path
- image_folder
- jason_path

## Adjustable Parameters

### SimCLR Script

- size in "TransformsCimCLR": image size after transformation, currently very small. Adjust according to your computational resources. 
- project_dim: size of the projection head's output. Tune based on complexity of the dataset. Potential area of improvement - other numbers were not tested. 
- lr in 'optimizer': learning rate for the model
- margin in "TripletMarginLoss": Sets how far apart the dissimilar images should be in the learned feature space. Potential area of improvement - other numbers were not tested.

### OneClassSVM Script 

- See documentation for OneClassSVM

### Anomaly Detect Script

- See documentation for DBScan

## Overview

### SimCLR Script

SimCLR (Simple Framework for Contrastive Learning of Visual Representation) is a framework that learns visual representation by maximizing agreement between differently augmented views of the same data example via a contranstive loss in the latent data space. It learns that similar images (augmentations of the same image) have similar encodings and different images have different encodings. ResNet50 was used as the encoder. 

Triplet Dataset (creates an anchor, positive image, and negative image) was used to generate the augementations.

Images are cropped with available bounding box data and analysis is done on the species level. A weighted sampler is used to offset the class imbalance in the dataset. 

### OneClassSVM Script

Bounding boxes are again used to crop the images. The features are extracted from the trained SimCLR model and OneClassSVM is used to learn each decision boundary in a higher dimensional space for feature vectors on a species specific level. Individual SVM models are saved for later use. 

### Anomaly Detect Script 

The prediction dataset does not have bounding boxes so a trained YOLO model is used for object detection. The bounding boxes produced by YOLO are used to crop images and then fed through the SimCLR and OneClassSVM process.

_Note: Untested code exists in the script to predict anomalies with YOLO species classification. If YOLO predicts a species that does not exist in the pretrained SVM models, then classify that image as an anomaly. This will not work with YOLO that has only been trained on the same image set as they will both only know the same set of species, but may be useful with a broad/pretrained YOLO model_

The feature space is visualized with t-SNE. At this point, **certain species were manually obscured from the visualization with opacity levels for readability. A species with a low opacity will not appear on the legend.**

DBScan using the t-SNE transformed features (for readability) is applied to find anomalies. Eps is also plotted for each run to evaluate parameter choice. 

## Additional Script to Generate Synthetic Anomalies 

Also included is a one-off script to generate synthetic anomalies. It requires a folder of images with backgrounds, and then images of the species that you wish to create anomalamous images of, and bounding box data. The script crops species, and randomly overlays it on a few different backgrounds in random places with random scaling and mild transparency.

## Misc Scripts

I included some scripts that were related to processesing images for YOLO, or created train/val folders. They were just preprocessing that aren't really related to machine learning and very specific to this image dataset but I included them just incase they were helpful. YOLO itself you run from command line - there isn't really a script for it. I used Ultranalytics YOLO v8. 