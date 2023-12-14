import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def filter_lines(file_path):
    filtered_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                num_boxes = int(parts[1])
                if num_boxes != 0:
                    filtered_lines.append(line.split()[0])
    return filtered_lines

# Specify the path to your text file
file_path = "labels.txt"

# Call the function to filter the lines
filtered_lines = filter_lines(file_path)

IMAGE_SIZE = (224, 224)  # You can adjust this size based on your model requirements

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    
    class_names = os.listdir(data_dir)
    counter = 0 
    for class_name in class_names:
        counter += 1
        if counter < 15:
            class_dir = os.path.join(data_dir, class_name)
            print(class_dir)
            for image_dir in os.listdir(class_dir):
                image_path_dir = os.path.join(class_dir, image_dir)
                image_names = os.listdir(image_path_dir)
                for image_name in image_names:
                    image_path = os.path.join(image_path_dir, image_name)
                    if image_name.endswith('.JPG'):
                        dir1 = os.path.join(class_name,image_dir)
                        dir2 = os.path.join(dir1,image_name)
                        image = cv2.imread(image_path)
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, IMAGE_SIZE)
                        image = image / 255.0  # Normalize the pixel values to the range [0, 1]
                        images.append(image)
                        if dir2 in filtered_lines:
                            labels.append(class_name)
                        else:
                            labels.append("no-detection")

    
    return np.array(images), np.array(labels)


images, labels = load_and_preprocess_data("missouri/Set1")

label_encoder = LabelEncoder()
label_encoder.fit(labels)
labels_encoded = label_encoder.transform(labels)
num_classes = len(label_encoder.classes_)
y = to_categorical(labels_encoded,num_classes)

# X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)


input_tensor = Input(shape=(*IMAGE_SIZE, 3))
base_model = EfficientNetB0(input_tensor=input_tensor, include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))

# Add your custom head on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)

# You can freeze some layers if needed
for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

class_names = label_encoder.classes_
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)




