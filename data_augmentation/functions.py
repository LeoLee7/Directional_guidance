# augmentation functions for image perturbation

import pandas as pd
import json
import numpy as np
import cv2
import os
from PIL import Image, ImageOps

# Remove deprecated numpy aliases for compatibility
np.float = float    
np.int = int   
np.object = object    
np.bool = bool    

def check_task(task: str):
    """Print the current task split (for debugging)."""
    print(f'{task}')

def zoom_in(img_name, zoom_param, img_folder):
    """
    Zoom in on the image by a given fraction.
    Args:
        img_name (str): Image filename.
        zoom_param (float): Fraction to zoom in (0 < zoom_param < 1).
        img_folder (str): Folder containing the image.
    Returns:
        PIL.Image: Zoomed image.
    """
    zoom_fraction = 1 - zoom_param
    img_path = os.path.join(img_folder, img_name)
    image_original = Image.open(img_path)
    img = ImageOps.exif_transpose(image_original) 
    width, height = img.size
    new_width = int(width * zoom_fraction)
    new_height = int(height * zoom_fraction)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((width, height), Image.ANTIALIAS)         
    return resized_img   

def offset(img_name, offset_param, keyword='move to left', crop_mode='cut', img_folder=None, mask_folder=None):
    """
    Apply an offset perturbation to the image based on the mask.
    Args:
        img_name (str): Image filename.
        offset_param (float): Offset fraction.
        keyword (str): Type of movement ('move to left', etc.).
        crop_mode (str or float): Cropping mode or custom scale.
        img_folder (str): Folder containing the image.
        mask_folder (str): Folder containing the mask image.
    Returns:
        np.ndarray or PIL.Image: Perturbed image.
    """
    assert img_folder is not None and mask_folder is not None, "img_folder and mask_folder must be provided."
    img_path = os.path.join(img_folder, img_name)
    img_original = cv2.imread(img_path)
    mask_path = os.path.join(mask_folder, img_name.replace('.jpg', '.png'))
    img_mask = cv2.imread(mask_path) // 255    
    indices = np.argwhere(img_mask[:, :, 1] == 1)

    img_height, img_width = img_original.shape[:2]
    mask_height, mask_width = img_mask.shape[:2]
    if img_height != mask_height or img_width != mask_width:
        raise ValueError('Image and Mask are not matched!')

    min_x_location = min(indices[:, 1])
    min_x_center_ratio = 0.5 * min_x_location / img_width
    max_x_location = max(indices[:, 1])
    max_x_ratio = max_x_location / img_width
    min_y_location = min(indices[:, 0])
    min_y_center_ratio = 0.5 * min_y_location / img_height
    max_y_location = max(indices[:, 0])
    max_y_ratio = max_y_location / img_height

    croped_image = None
    if keyword == 'move to left':
        offset_fraction = offset_param
        crop_center_ratio_x = 0.5 - offset_fraction * (0.5 - min_x_center_ratio)
        crop_center_ratio_y = 0.5
        crop_center = [crop_center_ratio_x, crop_center_ratio_y]
        if crop_mode == 'auto':
            scale_ratio = 2 * crop_center_ratio_x
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif isinstance(crop_mode, float):
            scale_ratio = float(crop_mode)
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif crop_mode == 'cut':
            right_boundary = max_x_location - offset_fraction * (max_x_location - min_x_location)
            croped_image = cut_offset_effect(img_path, right_boundary)
    elif keyword == 'move to right':
        offset_fraction = offset_param
        crop_center_ratio_x = 0.5 + offset_fraction * (0.5 * max_x_ratio)
        crop_center_ratio_y = 0.5
        crop_center = [crop_center_ratio_x, crop_center_ratio_y]
        if crop_mode == 'auto':
            scale_ratio = 1 - offset_fraction * max_x_ratio
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif isinstance(crop_mode, float):
            scale_ratio = float(crop_mode)
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif crop_mode == 'cut':
            left_boundary = min_x_location + offset_fraction * (max_x_location - min_x_location)
            croped_image = cut_offset_effect(img_path, left_boundary, mode='cut_left')
    elif keyword == 'move to up':
        offset_fraction = offset_param
        crop_center_ratio_y = 0.5 - offset_fraction * (0.5 - min_y_center_ratio)
        crop_center_ratio_x = 0.5
        crop_center = [crop_center_ratio_x, crop_center_ratio_y]
        if crop_mode == 'auto':
            scale_ratio = 2 * crop_center_ratio_y
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif isinstance(crop_mode, float):
            scale_ratio = float(crop_mode)
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif crop_mode == 'cut':
            down_boundary = max_y_location - offset_fraction * (max_y_location - min_y_location)
            croped_image = cut_offset_effect(img_path, down_boundary, mode='cut_down')
    elif keyword == 'move to down':
        offset_fraction = offset_param
        crop_center_ratio_y = 0.5 + offset_fraction * (0.5 * max_y_ratio)
        crop_center_ratio_x = 0.5
        crop_center = [crop_center_ratio_x, crop_center_ratio_y]
        if crop_mode == 'auto':
            scale_ratio = 1 - offset_fraction * max_y_ratio
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif isinstance(crop_mode, float):
            scale_ratio = float(crop_mode)
            croped_image = zoom_in_offset_effect(img_path, scale_ratio, crop_center)
        elif crop_mode == 'cut':
            up_boundary = min_y_location + offset_fraction * (max_y_location - min_y_location)
            croped_image = cut_offset_effect(img_path, up_boundary, mode='cut_up')
    else:
        raise ValueError(f"Unknown keyword: {keyword}")
    return croped_image

def zoom_in_offset_effect(img_path, zoom_factor, center_ratio):
    """
    Zoom in on an image at a specific center.
    Args:
        img_path (str): Path to the image.
        zoom_factor (float): Zoom factor.
        center_ratio (list): [x_ratio, y_ratio] center.
    Returns:
        PIL.Image: Zoomed image.
    """
    image_original = Image.open(img_path)
    img = ImageOps.exif_transpose(image_original) 
    original_width, original_height = img.size
    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)
    center_x = int(center_ratio[0] * original_width)
    center_y = int(center_ratio[1] * original_height)
    left = max(0, center_x - new_width // 2)
    top = max(0, center_y - new_height // 2)
    right = min(original_width, left + new_width)
    bottom = min(original_height, top + new_height)
    cropped_img = img.crop((left, top, right, bottom))
    zoomed_img = cropped_img.resize((original_width, original_height), Image.ANTIALIAS)
    return zoomed_img

def cut_offset_effect(img_path, boundary, mode='cut_right'):
    """
    Cut the image at a given boundary.
    Args:
        img_path (str): Path to the image.
        boundary (int): Pixel boundary.
        mode (str): Which side to cut ('cut_right', 'cut_left', 'cut_up', 'cut_down').
    Returns:
        np.ndarray: Cropped image.
    """
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[:2]
    boundary = int(boundary)
    if mode == 'cut_right':
        assert boundary <= original_width, 'Wrong boundary setting!'
        cropped_img = img[:, :boundary]
    elif mode == 'cut_left':
        assert boundary <= original_width, 'Wrong boundary setting!'
        cropped_img = img[:, boundary:]
    elif mode == 'cut_up':
        assert boundary <= original_height, 'Wrong boundary setting!'
        cropped_img = img[boundary:, :]
    elif mode == 'cut_down':
        assert boundary <= original_height, 'Wrong boundary setting!'
        cropped_img = img[:boundary, :]
    else:
        raise ValueError('Invalid mode specified!')
    return cropped_img