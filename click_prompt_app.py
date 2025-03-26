import os
import sys
import numpy as np
import nibabel as nib
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.models import load_model

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QWidget, QSlider)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

import gymnasium as gym
from gymnasium import spaces

def find_best_slice(lesion_data):
    slice_sums = [np.sum(lesion_data[:, :, i]) for i in range(lesion_data.shape[2])]
    return np.argmax(slice_sums)

class CropEnvironment(gym.Env):
    def __init__(self, image, initial_point, lesion_mask=None, crop_size=(10, 10, 1)):
        super().__init__()
        self.image = image
        self.lesion_mask = lesion_mask
        self.crop_size = crop_size
        self.img_shape = image.shape
        self.initial_point = initial_point

        self.observation_space = spaces.Box(low=0, high=1, shape=(16, 256, 256, 1), dtype=np.float32)
        self.action_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.int32)

        self.model = load_model('/Users/lynnkaram/Desktop/saved_models/model_epoch_20.keras')
        
        self.velocity = np.zeros(3)
        self.momentum = 0.9
        
        self.reset()

    def reset(self):
        self.i, self.j, self.k = self.initial_point
        self.velocity = np.zeros(3)
        cropped_image = self.crop_volume(self.image, self.i, self.j, self.k, self.crop_size)
        resized_cropped_image = self.resize_to_model_input(cropped_image)
        self.current_observation = np.expand_dims(resized_cropped_image, axis=-1)
        return self.current_observation

    def step(self, action):
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * action
        
        new_i = np.clip(self.i + int(self.velocity[0]), 0, self.img_shape[0] - self.crop_size[0])
        new_j = np.clip(self.j + int(self.velocity[1]), 0, self.img_shape[1] - self.crop_size[1])
        new_k = np.clip(self.k + int(self.velocity[2]), 0, self.img_shape[2] - self.crop_size[2])

        cropped_image = self.crop_volume(self.image, new_i, new_j, new_k, self.crop_size)
        resized_cropped_image = self.resize_to_model_input(cropped_image)
        self.current_observation = np.expand_dims(resized_cropped_image, axis=-1)

        self.i, self.j, self.k = new_i, new_j, new_k

        predicted_score = self.model.predict(np.expand_dims(self.current_observation, axis=0), verbose=0).flatten()[0]
        
        proximity_reward = 0
        if self.lesion_mask is not None:
            current_region = self.lesion_mask[new_i:new_i+self.crop_size[0], 
                                            new_j:new_j+self.crop_size[1], 
                                            new_k:new_k+self.crop_size[2]]
            proximity_reward = np.sum(current_region) / current_region.size
            
            if proximity_reward > 0:
                padded_region = np.pad(current_region, 1)
                edges = np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[:-2, 1:-1, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[2:, 1:-1, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[1:-1, :-2, 1:-1]) + \
                       np.sum(padded_region[1:-1, 1:-1, 1:-1] != padded_region[1:-1, 2:, 1:-1])
                if edges > 0:
                    proximity_reward *= 1.2
        
        reward = 0.4 * predicted_score + 0.6 * proximity_reward
        done = False

        return self.current_observation, reward, done, {
            'position': (self.i, self.j, self.k), 
            'score': reward
        }

    def crop_volume(self, volume, i, j, k, crop_size=(10, 10, 1)):
        return volume[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]

    def resize_to_model_input(self, cropped_image, target_shape=(16, 256, 256)):
        depth_factor = target_shape[0] / cropped_image.shape[-1]
        width_factor = target_shape[1] / cropped_image.shape[0]
        height_factor = target_shape[2] / cropped_image.shape[1]
        resized_image = ndimage.zoom(cropped_image, (width_factor, height_factor, depth_factor), order=1)
        return resized_image

class CancerDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Interactive Cancer Region Detection')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout.addWidget(self.canvas)

        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel)

        load_button = QPushButton('Load NIfTI Image')
        load_button.clicked.connect(self.load_image)
        control_panel.addWidget(load_button)

        self.label_next_to_load_button = QLabel("Load an image to get started")
        control_panel.addWidget(self.label_next_to_load_button)

        self.slice_label = QLabel('Current Slice: 0')
        control_panel.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(15)
        self.slice_slider.valueChanged.connect(self.update_slice)
        control_panel.addWidget(self.slice_slider)

        self.nifti_image = None
        self.lesion_mask = None
        self.current_slice = 0
        self.rl_environment = None
        self.cancer_regions = []

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open NIfTI Image', '', 'NIfTI Files (*.nii.gz)')
        if file_path:
            nifti_img = nib.load(file_path)
            self.nifti_image = nifti_img.get_fdata()
            
            mask_path = os.path.join(os.path.dirname(file_path), 'l_a1.nii.gz')
            if os.path.exists(mask_path):
                lesion_img = nib.load(mask_path)
                self.lesion_mask = lesion_img.get_fdata()
                best_slice = find_best_slice(self.lesion_mask)
            else:
                self.lesion_mask = None
                best_slice = 0
            
            target_shape = (256, 256, 16)
            width_factor = target_shape[0] / self.nifti_image.shape[0]
            height_factor = target_shape[1] / self.nifti_image.shape[1]
            depth_factor = target_shape[2] / self.nifti_image.shape[2]
            
            self.nifti_image = ndimage.zoom(self.nifti_image, 
                                          (width_factor, height_factor, depth_factor), 
                                          order=1)
            
            if self.lesion_mask is not None:
                self.lesion_mask = ndimage.zoom(self.lesion_mask,
                                              (width_factor, height_factor, depth_factor),
                                              order=1)
            
            self.current_slice = int(best_slice * depth_factor)
            self.slice_slider.setValue(self.current_slice)
            
            self.cancer_regions = []
            self.label_next_to_load_button.setText("Image loaded. Click to detect cancer regions.")
            self.display_image()

    def update_slice(self, value):
        self.current_slice = value
        self.display_image()

    def on_click(self, event):
        if self.nifti_image is None or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        initial_point = (x, y, self.current_slice)
        
        self.rl_environment = CropEnvironment(self.nifti_image, initial_point, self.lesion_mask)
        self.cancer_regions = []
        max_steps = 200
        explored_positions = set()
        
        for step in range(max_steps):
            if step == 0:
                action = np.random.randint(-5, 5, size=3)
            else:
                angle = 2 * np.pi * step / 20
                radius = step / 10
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                action = np.array([dx, dy, 0])
            
            _, reward, _, info = self.rl_environment.step(action)
            position = info['position']
            pos_key = (position[0] // 5, position[1] // 5)
            
            if reward > 0.05 and pos_key not in explored_positions:
                self.cancer_regions.append(position)
                explored_positions.add(pos_key)

        self.display_image()

def display_image(self):
        if self.nifti_image is None:
            return
          
        self.figure.clear()
        ax = self.figure.add_subplot(111)
    
        # Display original image
        ax.imshow(self.nifti_image[:, :, self.current_slice], cmap='gray')
    
        # Show lesion mask with a thick blue line
        if self.lesion_mask is not None:
            lesion_slice = self.lesion_mask[:, :, self.current_slice]
            if np.any(lesion_slice > 0):
                ax.contour(lesion_slice, levels=[0.5], colors='blue', linewidths=3)
    
        # Create a blank prediction mask
        prediction_mask = np.zeros_like(self.nifti_image[:, :, self.current_slice], dtype=np.uint8)
    
        # Fill prediction mask based on cancer regions
        box_size = 10
        for region in self.cancer_regions:
            x, y, z = region
            if z == self.current_slice:
                prediction_mask[x:x+box_size, y:y+box_size] = 1
    
        # Display single red lesion mask 
        if np.any(prediction_mask):
            ax.contourf(prediction_mask, levels=[0.5, 1], colors='red', alpha=0.4)
    
            if self.lesion_mask is not None:
                lesion_slice = self.lesion_mask[:, :, self.current_slice]
                intersection = np.sum(np.logical_and(prediction_mask, lesion_slice > 0))
                union = np.sum(np.logical_or(prediction_mask, lesion_slice > 0))
                if union > 0:
                    overlap_percentage = (intersection / union) * 100
                    ax.text(10, 20, f'Overlap: {overlap_percentage:.1f}%', 
                            color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    
        self.canvas.draw()
        self.slice_label.setText(f'Current Slice: {self.current_slice}')


def main():
    app = QApplication(sys.argv)
    ex = CancerDetectionApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
