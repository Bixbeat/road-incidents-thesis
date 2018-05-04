import string
import random

import os
from PIL import Image

def create_dir_if_not_exist(directory):
    """Creates a directory if the path does not yet exist.

    Args:
        directory (string): The directory to create.
    """          
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def generate_random_filename(length=10):
    # https://www.pythoncentral.io/python-snippets-how-to-generate-random-string/
    allchar = string.ascii_letters + string.digits
    return("".join(random.choice(allchar) for x in range(length)))