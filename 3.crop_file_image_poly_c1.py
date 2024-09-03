#!/usr/env python
# -*- coding:utf-8 -*-
# author:qianqian time:6/11/2024

import os
from PIL import Image, ImageDraw


def crop_non_rectangular(input_image_path, output_image_path, mask_coordinates):
    """
    Crop an image using a non-rectangular mask and save the cropped image.

    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the cropped image.
    :param mask_coordinates: A list of (x, y) tuples defining the polygon mask.
    """
    # Ensure the polygon is closed by adding the first point to the end
    if mask_coordinates[0] != mask_coordinates[-1]:
        mask_coordinates.append(mask_coordinates[0])

    # Open an image file
    with Image.open(input_image_path) as img:
        # Create a mask image with the same size as the input image, filled with black (transparent)
        mask = Image.new('L', img.size, 0)
        # Create a drawing context for the mask
        draw = ImageDraw.Draw(mask)
        # Draw a polygon on the mask with white (opaque)
        draw.polygon(mask_coordinates, fill=255)
        # Apply the mask to the input image
        cropped_img = Image.composite(img, Image.new('RGB', img.size), mask)

        # Further crop the image to remove black borders
        bbox = cropped_img.getbbox()
        final_cropped_img = cropped_img.crop(bbox)

        # Save the final cropped image
        final_cropped_img.save(output_image_path)
        print(f"Cropped image saved at {output_image_path}")


def process_images(input_folder, output_folder, mask_coordinates):
    """
    Process all .jpg images in the input_folder, apply the non-rectangular crop, and save to the output_folder.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder to save cropped images.
    :param mask_coordinates: A list of (x, y) tuples defining the polygon mask.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f"cropped_{filename}")
            crop_non_rectangular(input_image_path, output_image_path, mask_coordinates)


# Merge A
input_folder = 'D:/Video_frame/video_frames_c'
output_folder = 'D:/Video_frame/video_frames_c/MA'
mask_coordinates = [
    (5796, 620),
    (5796, 853),
    (1948, 1009),
    (948, 1124),
    (948, 938),
    (3037, 736)
]


process_images(input_folder, output_folder, mask_coordinates)
