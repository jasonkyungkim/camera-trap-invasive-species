#!/usr/bin/env bash

# Change the format of the animl data folder to be compatible with HF
python3 restructure_animl_jldp_folder.py \
   '../../../animl_dp_data/' \
   'jldp-animl-images-raw/' \
   'jldp-animal-images-uncropped-processed-512/' \
   'jldp-animl-cct.json'\
   --label-multiclass-images-with 'majority'\
   512\
   512

python3 create_hf_dataset.py \
   '../../../animl_dp_data/jldp-animal-images-uncropped-processed-512/' \
   '../../../animl_dp_data/hf_dataset_jldp_animl_512.hf'
