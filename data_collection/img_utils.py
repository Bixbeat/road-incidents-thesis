#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""

from PIL import Image
import os

def delete_equal_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for image1 in files:
            img_path_1 = os.path.join(root, image1)
            for image2 in files:
                img_path_2 = os.path.join(root, image2)
                
                remove_path_from_list = del_image_if_equal(img_path_1, img_path_2)
                
                if remove_path_from_list == True:
                    files.remove(image2)

def del_image_if_equal(img_path_1, img_path_2):
    img_1 = Image.open(img_path_1)
    img_2 = Image.open(img_path_2)
    remove_path = False

    if img_1 == img_2 and img_path_1 != img_path_2:
        print(f'{img_path_1} is equal to {img_path_2}')
        os.remove(img_path_2)
        remove_path = True
    
    return remove_path

if __name__ == '__main__':
    delete_equal_images('/home/alex/Documents/test')
    # delete_equal_images('/media/alex/A4A034E0A034BB1E/incidents-thesis/data/snow')