"""
From cropped images, create and save a Tensorflow dataset
"""

import argparse
import tensorflow as tf
import keras
import pickle

def main(img_directory: str,
         ds_directory: str,
         batch_size: int,
         image_height: int,
         image_width: int,
         validation_split: float) -> None:
    seed = 123
    image_size = (image_height, image_width)

    train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
        img_directory,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="both",
        crop_to_aspect_ratio=False,
        )
    
    train_ds.save(ds_directory+"/trainval_ds")
    test_ds.save(ds_directory+"/test_ds")
    # Class names unfortunately not saved with the dataset, need to save separately
    # Save one layer up to not screw with the tf datasets
    class_name_directory = "/".join(ds_directory.split("/")[:-1])
    with open(class_name_directory+'/class_names.pkl', 'wb') as f:
        pickle.dump(train_ds.class_names, f)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create TF dataset from MD crops.')
    parser.add_argument(
        'img_directory',
        help='path to directory where images are stored under label subdirectories')
    parser.add_argument(
        'ds_directory',
        help='path to directory where datasets will be saved')
    parser.add_argument(
        'batch_size', type=int,
        help='batch size in dataset')
    parser.add_argument(
        'image_height', type=int,
        help='int height to cast images to')
    parser.add_argument(
        'image_width', type=int,
        help='int width to cast images to')
    parser.add_argument(
        'validation_split', type=float,
        help='fraction of data to reserve for the test set')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(img_directory=args.img_directory,
         ds_directory=args.ds_directory,
         batch_size=args.batch_size,
         image_height=args.image_height,
         image_width=args.image_width,
         validation_split=args.validation_split)