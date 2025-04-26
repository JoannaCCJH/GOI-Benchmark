from datasets import load_dataset
from datasets import load_dataset_builder

# dataset = load_dataset("GaussianWorld/scannetpp_v1_mcmc_1.5M_3dgs")

# print(dataset)

# ds_builder = load_dataset_builder("GaussianWorld/scannetpp_v1_mcmc_1.5M_3dgs")

# print(ds_builder.info.description)
# print(ds_builder.info.features)

from huggingface_hub import hf_hub_download
import os

# Set the repository and folder details
# repo_id = "GaussianWorld/scannetpp_v1_mcmc_1.5M_3dgs"
# folder_name = "09c1414f1b"
# local_dir = f"./{folder_name}"

repo_id = "GaussianWorld/scannetpp_v1_val_subset"
folder_name = "09c1414f1b/dslr/undistorted_images"
local_dir = f"{folder_name}"

# # Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# # First, list all files in the repository to find ones in your folder
from huggingface_hub import list_repo_files

# all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# # Filter for files only in the folder you want
# folder_files = [file for file in all_files if file.startswith(folder_name)]

# # Download each file in the folder
# for file_path in folder_files:
#     file_name = os.path.basename(file_path)
#     output_path = os.path.join(local_dir, file_name)
    
#     # Download the file
#     downloaded_path = hf_hub_download(
#         repo_id=repo_id,
#         filename=file_path,
#         repo_type="dataset"
#     )
    
#     # If you want to copy it to your specified directory
#     import shutil
#     shutil.copy(downloaded_path, output_path)
#     print(f"Downloaded: {file_path} to {output_path}")
    
# # Set the repository and folder details
# repo_id = "GaussianWorld/scannetpp_v1_mcmc_1.5M_3dgs"
# folder_name = "09c1414f1b"
# local_dir = f"./{folder_name}"

# # Create the directory if it doesn't exist
# os.makedirs(local_dir, exist_ok=True)



all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# Filter for files only in the folder you want
folder_files = [file for file in all_files if file.startswith(folder_name)]

# Download each file in the folder
for file_path in folder_files:
    file_name = os.path.basename(file_path)
    output_path = os.path.join(local_dir, file_name)
    
    # Download the file
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset"
    )
    
    # If you want to copy it to your specified directory
    import shutil
    shutil.copy(downloaded_path, output_path)
    print(f"Downloaded: {file_path} to {output_path}")