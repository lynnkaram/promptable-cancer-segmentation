import os 
import numpy as np  
import nibabel as nib  
from scipy import ndimage  
import tensorflow as tf  
from tensorflow.keras.models import load_model  

def calculate_dice_score(prediction_mask, ground_truth_mask):
    """Calculate Dice similarity coefficient."""  
    intersection = np.sum(prediction_mask * ground_truth_mask)  
    union = np.sum(prediction_mask) + np.sum(ground_truth_mask)  
    if union == 0:  
        return 1.0  
    return 2.0 * intersection / union  # Calculate Dice coefficient: 2*intersection/union

def find_best_slice(lesion_data):
    # Creates a list of sums of pixel/voxel value in each 2D slice of the 3D medical image 
    # and returns index of slice with maximum sum. Slice with highest sum = slice where lesion appears most
    slice_sums = [np.sum(lesion_data[:, :, i]) for i in range(lesion_data.shape[2])]
    return np.argmax(slice_sums)  # Return index of slice with largest sum

def find_click_point(lesion_mask_slice):
    """Find a good point to click within the lesion mask."""
    if lesion_mask_slice is None or not np.any(lesion_mask_slice > 0):  # Check if mask is empty
        return None
    # Label connected components = regions of pixels that are touching each other in the binary mask
    labeled_mask, num_features = ndimage.label(lesion_mask_slice > 0) #binary mask where 1 = lesion, 0 = background
    if num_features == 0:  # If no components found
        return None
    # Calculate size of each component/lesion 
    sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)] 
    largest_component = np.argmax(sizes) + 1  # Find largest component                
    component_mask = labeled_mask == largest_component                                            
    # Find center of mass of largest component
    cy, cx = ndimage.center_of_mass(component_mask) #cx and cy are averages of x and y coordinates
    return int(cx), int(cy)  # Return integer coordinates of center point

class CropEnvironment: 
    def __init__(self, image, initial_point, lesion_mask=None, crop_size=(10, 10, 6)):
        self.image = image  # Store input image
        self.lesion_mask = lesion_mask  # Store lesion mask
        self.crop_size = crop_size  # Store crop window size
        self.img_shape = image.shape  # Store image dimensions
        self.initial_point = initial_point  # Store starting point
        # Load pre-trained model from specified path
        self.model = load_model('/home/lynn/model_epoch_20.keras') 
        self.velocity = np.zeros(3)  # Initialize velocity vector
        self.momentum = 0.9  # Set momentum coefficient
        self.reset()  # Initialise environment

    def reset(self): 
        # Reset position to initial point
        self.i, self.j, self.k = self.initial_point
        self.velocity = np.zeros(3)  # Reset velocity
        # Crop image at current position
        cropped_image = self.crop_volume(self.image, self.i, self.j, self.k, self.crop_size)
        # Resize cropped image to model input size
        resized_cropped_image = self.resize_to_model_input(cropped_image)
        # Add channel dimension and store as current observation
        self.current_observation = np.expand_dims(resized_cropped_image, axis=-1)
        return self.current_observation

    def step(self, action):
        # Update velocity using momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * action    
        # Calculate new position with bounds checking
        new_i = np.clip(self.i + int(self.velocity[0]), 0, self.img_shape[0] - self.crop_size[0])
        new_j = np.clip(self.j + int(self.velocity[1]), 0, self.img_shape[1] - self.crop_size[1])
        new_k = np.clip(self.k + int(self.velocity[2]), 0, self.img_shape[2] - self.crop_size[2])
        # Crop image at new position
        cropped_image = self.crop_volume(self.image, new_i, new_j, new_k, self.crop_size)
        # Resize cropped image
        resized_cropped_image = self.resize_to_model_input(cropped_image)
        # Update current observation
        self.current_observation = np.expand_dims(resized_cropped_image, axis=-1)
        # Update position
        self.i, self.j, self.k = new_i, new_j, new_k
        # Get model prediction for current observation
        predicted_score = self.model.predict(np.expand_dims(self.current_observation, axis=0), verbose=0).flatten()[0]     
        proximity_reward = 0
        if self.lesion_mask is not None:
            # Extract current region from lesion mask
            current_region = self.lesion_mask[new_i:new_i+self.crop_size[0], 
                                            new_j:new_j+self.crop_size[1], 
                                            new_k:new_k+self.crop_size[2]]
            # Calculate proportion of lesion in current region
            proximity_reward = np.sum(current_region) / current_region.size       
            if proximity_reward > 0:
                # Pad region for edge detection
                padded_region = np.pad(current_region, 1)
                # Calculate edges by comparing with neighboring voxels
                edges = np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[:-2, 1:-1, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[2:, 1:-1, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[1:-1, :-2, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[1:-1, 2:, 1:-1])                    
                # Boost reward if edges are present
                if edges > 0:
                    proximity_reward *= 1.2  #(20%) boost if the crop contains edges of the legion     
        # Calculate final reward as weighted sum of model prediction and proximity
        # Reward is a combination of 2 factors with different weights: predicted_score comes from the pre-trained model's prediction (classifier)
        # and proximity_reward  measures how close we are to the lesion
        reward = 0.4 * predicted_score + 0.6 * proximity_reward
        # Return observation, reward, done flag, and info dictionary
        return self.current_observation, reward, False, {
            'position': (self.i, self.j, self.k), 
            'score': reward
        }
    def crop_volume(self, volume, i, j, k, crop_size=(10, 10, 6)):
        # Extract subset of volume using specified coordinates and crop size
        return volume[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]
    def resize_to_model_input(self, cropped_image, target_shape=(16, 256, 256)):
        # This function resizes the cropped volume to match what the model expects as input
        depth_factor = target_shape[0] / cropped_image.shape[-1] 
        width_factor = target_shape[1] / cropped_image.shape[0]  
        height_factor = target_shape[2] / cropped_image.shape[1] 
        # Resize image using calculated factors
        resized_image = ndimage.zoom(cropped_image, (width_factor, height_factor, depth_factor), order=1)
        return resized_image



def process_single_image(image_path):
    """Process a single image and return its Dice score."""
    nifti_img = nib.load(image_path)
    nifti_image = nifti_img.get_fdata() 
    mask_path = os.path.join(os.path.dirname(image_path), 'l_a1.nii.gz')
    if not os.path.exists(mask_path):
        print(f"No lesion mask found for {image_path}")
        return None       
    # Load the lesion mask (ground truth)
    lesion_img = nib.load(mask_path)
    lesion_mask = lesion_img.get_fdata()
    # Reshape the nifti image to the lesion_mask.shape 
    nifti_image = ndimage.zoom(nifti_image, 
                          (lesion_mask.shape[0]/nifti_image.shape[0],
                           lesion_mask.shape[1]/nifti_image.shape[1], 
                           lesion_mask.shape[2]/nifti_image.shape[2]), 
                          order=1)
    # Define target dimensions for resizing
    target_shape = (256, 256, 16)
    # Calculate scaling factors for each dimension
    width_factor = target_shape[0] / nifti_image.shape[0]
    height_factor = target_shape[1] / nifti_image.shape[1]
    depth_factor = target_shape[2] / nifti_image.shape[2]   
    # Resize both image and mask to target shape
    nifti_image = ndimage.zoom(nifti_image, (width_factor, height_factor, depth_factor), order=1)
    lesion_mask = ndimage.zoom(lesion_mask, (width_factor, height_factor, depth_factor), order=1)    
    # Find the slice with the largest lesion area
    best_slice = find_best_slice(lesion_mask)
    # Find the center point of the largest lesion in that slice
    click_point = find_click_point(lesion_mask[:, :, best_slice])    
    if click_point is None: # this is if no lesion was found
        print(f"No suitable click point found in {image_path}")
        return None
    # Create starting point from click coordinates and best slice
    initial_point = (click_point[0], click_point[1], best_slice)
    # Initialize the environment with the image and mask
    env = CropEnvironment(nifti_image, initial_point, lesion_mask)
    # Create empty mask for predictions (same size as input image)
    prediction_mask = np.zeros_like(nifti_image)
    max_steps = 200
    explored_positions = set()  
    # Main loop for exploring the image
    for step in range(max_steps):
        # For first step, move randomly
        if step == 0:
            action = np.random.randint(-5, 5, size=3)
        else:
            # Spiral search pattern
            angle = 2 * np.pi * step / 20  # Converts steps into angles
            radius = step / 10             # Radius grows with each step
            dx = radius * np.cos(angle)    # X coordinate on spiral
            dy = radius * np.sin(angle)    # Y coordinate on spiral
            action = np.array([dx, dy, 0])  # No movement in Z direction
        # Take a step in the environment
        _, reward, _, info = env.step(action)
        position = info['position'] # get current position
        pos_key = (position[0] // 5, position[1] // 5) #Group positions into 5x5 grid cells                          
        # If reward is significant and position hasn't been explored (avoid double counting)
        if reward > 0.05 and pos_key not in explored_positions:
            x, y, z = position
            prediction_mask[x:x+10, y:y+10, z:z+1] = 1 
            explored_positions.add(pos_key)    
    # Apply 3D Gaussian smoothing to the prediction mask
    sigma = [4, 4, 0.8]  
    prediction_mask = ndimage.gaussian_filter(prediction_mask, sigma=sigma)
    # Threshold the smoothed mask to create binary prediction
    threshold = 0.2  
    prediction_mask = (prediction_mask > threshold).astype(float)
    # Calculate Dice score
    dice_score = calculate_dice_score(prediction_mask, lesion_mask)
    return dice_score



def main():
    # Directory containing patient folders
    base_dir = os.path.expanduser('/home/lynn/image_with_masks')
    # Save output file 
    output_file = os.path.expanduser('/home/lynn/gaus_4_4_8.txt')
    dice_scores = []
    patient_numbers = []
    contents = os.listdir(base_dir)
    processed_count = 0
    # Create/clear the output file at the start
    with open(output_file, "w") as f:
        print("Patient_Number, Dice_Score", file=f)
    # Iterate through 40 patient directories
    for patient_folder in contents:
        if processed_count >= 40:
            break
        patient_path = os.path.join(base_dir, patient_folder)
        print(f"\nChecking patient folder: {patient_folder}")     
        patient_files = os.listdir(patient_path)
        nii_files = [f for f in patient_files if f.endswith('.nii.gz')]
        image_files = [f for f in nii_files if not f.startswith('l_')]               
        image_file = image_files[0]
        image_path = os.path.join(patient_path, image_file)
        mask_path = os.path.join(patient_path, 'l_a1.nii.gz')       
        if not os.path.exists(mask_path):
            print(f"No lesion mask found for {patient_folder}")
            continue
        dice_score = process_single_image(image_path)
        dice_scores.append(dice_score)
        patient_numbers.append(patient_folder)
        processed_count += 1
        # Append each result to the file as we go
        with open(output_file, "a") as f:
            print(f"{patient_folder},{dice_score:.4f}", file=f)
            
        print(f"Dice score for {patient_folder}: {dice_score:.4f}")
    # Calculate and print statistics
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        print("\nResults Summary:")
        print(f"Number of patients processed: {len(dice_scores)}")
        print(f"Mean Dice score: {mean_dice:.4f}")
        print(f"Standard deviation: {std_dice:.4f}")
    else:
        print("\nNo images were successfully processed")
    print(f"\nResults have been saved to: {output_file}")



main()




    


