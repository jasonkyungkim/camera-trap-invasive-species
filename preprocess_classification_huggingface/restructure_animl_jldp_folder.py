import os
import json
import shutil
from tqdm import tqdm
import argparse
from collections import Counter
from PIL import Image

def copy_paste_resize_img_file(curr_file_path: str, 
                               new_file_dir: str,
                               file_name: str,
                               new_img_width: int,
                               new_img_height: int):
    # Copying image into HF style folder structure
    if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)
    try:
        # Copy
        shutil.copy(curr_file_path, new_file_dir)
        # Resize
        image = Image.open(new_file_dir+file_name)
        curr_image_width, curr_image_height = image.size
        if new_img_width == 0:
            img_width = curr_image_width
        else:
            img_width = new_img_width
        if new_img_height == 0:
            img_height = curr_image_height
        else:
            img_height = new_img_height
        new_image = image.resize((img_width, img_height))
        new_image.save(new_file_dir+file_name)
        return 1
    except FileNotFoundError as e:
        print(e)
        return 0

def main(path_to_data_root: str,
         raw_data_folder: str,
         processed_data_folder: str,
         cct_json_filename: str,
         crop_to_bbox: bool,
         uncropped_multi_annotation_label: str,
         img_width: int,
         img_height: int,):
    
    accept_multi_annotation_actions = ["all", "majority"]
    if uncropped_multi_annotation_label not in accept_multi_annotation_actions:
        raise ValueError(
            "Acceptable ways to treat uncropped images with multiple labels are {}"
            .format(accept_multi_annotation_actions)
            )
    
    f = open(path_to_data_root + cct_json_filename)
    dpcct = json.load(f)

    # Convert list of dicts to a single dict for ease of use
    label_mapping = {}
    for d in dpcct['categories']:
        label_mapping[d['id']] = d['name']

    image_counter = 0
    annotation_counter = 0

    metadata_json = []
    for i in tqdm(range(len(dpcct['images']))):
        image_dict = dpcct['images'][i]

        img_id = image_dict['id'] # Matches to annotations

        # Get filepath and jpeg name
        # colons in filenames, replace before te split
        file_name_list = image_dict['file_name'].replace(":","_").split('/')
        file_name = file_name_list[-1]
        # For copying
        curr_file_path = path_to_data_root + raw_data_folder + '/'.join(file_name_list[1:])
        # Some file paths have spaces in, which are filled with _ in reality. Fix here
        curr_file_path = curr_file_path.replace(" ","_")

        # Match annotations to image
        # If more than one animal in image may have more annotations than images
        matching_annotations = filter(
            lambda d: d['image_id'] == img_id, dpcct['annotations']
            )
        
        labels = []
        bboxs = []
        categories = []
        # Updating metadata list
        new_line_dict = {}
        new_line_dict["file_name"] = file_name
        new_line_dict["text"] = " ".join([image_dict['datetime'], image_dict['location']])

        for matching_annotation in matching_annotations:
            label = label_mapping[matching_annotation['category_id']]
            labels.append(label)

            # bbox needs to be nested in the jsonl
            # if already nested then iterate and add to bboxs, 
            # otherwise just append the single list
            if any(isinstance(i, list) for i in matching_annotation['bbox']):
                for sub_list in matching_annotation['bbox']:
                    bboxs.append(sub_list)
            else:
                bboxs.append(matching_annotation['bbox'])
            # Category needs to be a flat list in the jsonl
            if isinstance(matching_annotation['category_id'], int):
                categories.append(matching_annotation['category_id'])
            elif any(isinstance(i, list) for i in matching_annotation['category_id']):
                for sub_list in matching_annotation['category_id']:
                    categories.extend(sub_list)
            else:
                categories.extend(matching_annotation['category_id'])

            annotation_counter += 1
        
        new_line_dict["objects"] = {"bbox":bboxs, "categories":categories}
        
        # Copy image into relevant folder(s) and resize if needed
        if not crop_to_bbox:
            if uncropped_multi_annotation_label == "majority":
                # Copy-paste
                majority_label = Counter(labels).most_common(1)[0][0]
                new_file_dir = path_to_data_root + processed_data_folder + 'train/'+ majority_label + '/'
                image_counter+=copy_paste_resize_img_file(curr_file_path, 
                                                          new_file_dir, 
                                                          file_name, 
                                                          img_width, 
                                                          img_height)
            elif uncropped_multi_annotation_label == "all":
                for label in labels:
                    new_file_dir = path_to_data_root + processed_data_folder + 'train/'+ label + '/'
                    image_counter+=copy_paste_resize_img_file(curr_file_path, 
                                                              new_file_dir, 
                                                              file_name, 
                                                              img_width, 
                                                              img_height)
        else:
            raise NotImplementedError

        # metadata_json.append(new_line_dict)


    # Writing metadata list to jsonl file
    # with open(path_to_data_root + processed_data_folder + 'train/metadata.jsonl', 'x') as outfile:
    #     for entry in metadata_json:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')

    print("Processed {} images and {} annotations".format(image_counter, annotation_counter))


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Restructure folder for huggingface imagefolder dataset')
    parser.add_argument(
        'path_to_data_root',
        help='path to root data directory, where json lives and '
        'data is then stored in raw and processed subdirectories')
    parser.add_argument(
        'raw_data_folder',
        help='name of directory where raw data is stored')
    parser.add_argument(
        'processed_data_folder',
        help='name of directory where processed data will be saved')
    parser.add_argument(
        'cct_json_filename',
        help='COCO camera traps json file for the dataset')
    parser.add_argument(
        '--crop-to-bbox', action='store_true', default=False,
        help='whether to crop images to provided bounding boxes when saving')
    parser.add_argument(
        '--label-multiclass-images-with', type=str, default='all',
        help='ignored if cropping. how to treat images with more than one class '
        'in when not cropping. If "all" then the image is duplciated under all '
        'the labels present. If majority then it is only included in the '
        'prevailing label in the image')
    parser.add_argument(
        'img_width', type=int, default=0,
        help='width to resize images to')
    parser.add_argument(
        'img_height', type=int, default=0,
        help='height to resize images to')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(path_to_data_root=args.path_to_data_root,
         raw_data_folder=args.raw_data_folder,
         processed_data_folder=args.processed_data_folder,
         cct_json_filename=args.cct_json_filename,
         crop_to_bbox=args.crop_to_bbox,
         uncropped_multi_annotation_label=args.label_multiclass_images_with,
         img_width=args.img_width,
         img_height=args.img_height)
        
    
