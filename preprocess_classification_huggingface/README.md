Steps to use this folder to train a Swin transformer:
1. Download jldp data from box
2. Use restructure_animl_jldp_folder.py to restructure into the format that Tensorflow or Huggingface requires
3. Run create_hf_dataset.py to make and save a Huggingface dataset (equivalent code for Tensorflow in create_uncropped_dataset.py but Huggingface worked better - TF saving and loading is less robust)
4. _The above steps can be done together in the preprocess.sh shell script_
5. Run train_swin.py. Note requires significant memory
