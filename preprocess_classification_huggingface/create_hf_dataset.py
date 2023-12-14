import argparse
from datasets import load_dataset, disable_caching
import shutil
import os
# import pandas as pd

def main(path_to_imgfolder: str,
         filepath_to_save: str):
    disable_caching()
    print("Loading data...")
    dataset = load_dataset("imagefolder", 
                           data_files={"train":path_to_imgfolder+"train/**"},
                           download_mode="force_redownload",
                           cache_dir="./remove_local_cache") 
    dataset.cleanup_cache_files()
    if os.path.isdir("./remove_local_cache"):
        shutil.rmtree("./remove_local_cache")
    print("Load complete")
    dataset.save_to_disk(filepath_to_save)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create HuggingFace dataset')
    parser.add_argument(
        'path_to_imgfolder',
        help='path to root folder to make dataset from, subdirs at '
        'this level are train, test, val')
    parser.add_argument(
        'filepath_to_save',
        help='filepath at which the hf dataset is saved')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(path_to_imgfolder=args.path_to_imgfolder,
         filepath_to_save=args.filepath_to_save)