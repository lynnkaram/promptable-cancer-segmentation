# Code to split the initial data into training, validating and test sets. This is to avoid randomising/using different images every time I want to use the classifier 

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_into_folders(main_directory, output_directory):
    # Split data into training (50%), validation (20%), and test (30%) sets
    
    subfolders = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]
    
    # Split subfolders into training, validation, and test
    train_subfolders, test_val_subfolders = train_test_split(subfolders, test_size=0.5, random_state=42, stratify=None)
    val_subfolders, test_subfolders = train_test_split(test_val_subfolders, test_size=0.6, random_state=42, stratify=None)
    
    # Define paths for train, validation, and test folders
    train_dir = os.path.join(output_directory, 'train')
    val_dir = os.path.join(output_directory, 'validation')
    test_dir = os.path.join(output_directory, 'test')

    # Create the directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy subfolders to the respective directories
    def copy_subfolders(subfolders, destination):
        for subfolder in subfolders:
            src_path = os.path.join(main_directory, subfolder)
            dest_path = os.path.join(destination, subfolder)
            shutil.copytree(src_path, dest_path)

    copy_subfolders(train_subfolders, train_dir)
    copy_subfolders(val_subfolders, val_dir)
    copy_subfolders(test_subfolders, test_dir)

    # Print summary of data split
    print(f"Data successfully split into train, validation, and test sets.")
    print(f"Train: {len(train_subfolders)} subfolders")
    print(f"Validation: {len(val_subfolders)} subfolders")
    print(f"Test: {len(test_subfolders)} subfolders")

# Example usage of the function on my desktop
main_data_path = os.path.expanduser("/Users/lynnkaram/Desktop/all_images_copy")
output_data_path = os.path.expanduser("~/Desktop/split_data")
split_data_into_folders(main_data_path, output_data_path)
