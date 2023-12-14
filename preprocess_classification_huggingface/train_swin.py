from datasets import load_from_disk
import evaluate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers.keras_callbacks import (
    KerasMetricCallback, 
    PushToHubCallback)
from transformers import (
    DefaultDataCollator, 
    AutoImageProcessor,
    TFAutoModelForImageClassification,
    pipeline)
import json
from huggingface_hub import login

# Get secrets
with open("../../secrets/secrets.json") as f:
    hf_token = json.load(f)["hf_token"]

model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" # pre-trained model from which to fine-tune
batch_size = 64 # batch size for training and evaluation
path_to_saved_dataset = '../../../animl_dp_data/hf_dataset_jldp_animl_512.hf'
trainval_test_split = 0.15
train_val_split = 0.15

dataset = load_from_disk(path_to_saved_dataset)

# Convert id to labels and back
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Get rid of the boycot class with only a single image
dataset["train"] = dataset["train"].filter(lambda example: example["label"] != label2id["boycot"])

# train_val + test 
trainval_test = dataset['train'].train_test_split(test_size=trainval_test_split, 
                                         stratify_by_column="label", 
                                         shuffle=True)
# Split the train + valid into train and valid
train_val = trainval_test['train'].train_test_split(test_size=train_val_split, 
                                                    stratify_by_column="label",
                                                    shuffle=True)

train_ds = train_val['train']
val_ds = train_val['test']
test_ds = trainval_test['test']

# Make smaller dataset for testing
train_ds = train_ds.select(np.random.choice(train_ds.num_rows, 1000, replace=False))

# Plot examples
tiles=9
idx_sample = np.random.choice(train_ds.num_rows,tiles)
plt.figure(figsize=(10, 10))
for i, idx in enumerate(idx_sample):
    image = np.array(train_ds[int(idx)]['image'])
    label = id2label[train_ds[int(idx)]['label']]
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.astype("uint8"))
    plt.title(label)
    plt.axis("off")
plt.savefig("train_examples.png")

# Instantiate image processor
image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

# Set up data augmentation and preprocessing
size = (image_processor.size["height"], image_processor.size["width"])

train_data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        keras.layers.RandomBrightness([-0.8,0.8]),
        keras.layers.RandomContrast(0.2)
    ],
    name="train_data_augmentation",
)

def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

def preprocess_valtest(example_batch):
    images = [
        convert_to_tf_tensor(image.convert("RGB")) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

# Apply transforms only when data loaded into RAM
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_valtest)
test_ds.set_transform(preprocess_valtest)

# Batches
data_collator = DefaultDataCollator(return_tensors="tf")

# Define metrics to compute
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision_score = precision.compute(predictions=predictions, references=labels)
    recall_score = recall.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)
    acc_score = accuracy.compute(predictions=predictions, references=labels)
    return {"precision": precision_score, 
            "recall": recall_score, 
            "f1": f1_score,
            "accuracy": acc_score }


# Set up Adam optimizer with a learning rate schedule 
# using a warmup phase followed by a linear decay
# SWIN paper uses AdamW optimizer, 300 epochs, cosine decay learning rate scheduler
# 20 epochs of linear warm-up, batch size of 1024, initial learning rate of 0.001,
# weight decay  0.05

epochs = 5
batch_size = 32
decay_steps = len(train_ds) * epochs
initial_learning_rate = 0.001
lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps, warmup_steps=0
)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_warmup_decayed_fn,
    weight_decay=0.05
)

# Load model
model = TFAutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes = True,
)

# converting our train dataset to tf.data.Dataset
tf_train_ds = train_ds.to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

tf_val_ds = val_ds.to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

tf_test_ds = test_ds.to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# loss and compile
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# callbacks
login(token=hf_token)
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_val_ds)
push_to_hub_callback = PushToHubCallback(
    output_dir="tnc-camera-trap-swin",
    tokenizer=image_processor,
    save_strategy="no",
    hub_token=hf_token
)
callbacks = [metric_callback, push_to_hub_callback]

# fit
model.fit(tf_train_ds, validation_data=tf_val_ds, epochs=epochs, callbacks=callbacks)

pass
# Get best model from hub ready to do inference
pipe = pipeline(
    task="image-classification",
    model="colliers95/tnc-camera-trap-swin"
)

task_evaluator = evaluate.evaluator("image-classification")
eval_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=tf_test_ds,
    metric=evaluate.combine(["precision", "recall", "f1", "accuracy"]),
    label_mapping=pipe.model.config.label2id
)


