"""
Sources:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/camvid_loader.py
    https://github.com/bodokaiser/piwise/blob/master/piwise/dataset.py
    https://github.com/bfortuner/pytorch_tiramisu/blob/master/camvid_dataset.py    
"""
import os, os.path
import numpy as np
import scipy.misc as misc
from shutil import copyfile

from PIL import Image

def get_normalize_params(all_image_filepaths, num_bands):
    """For a set of image filepaths, returns the mean
    and stdev of all bands of all images in the set
    TODO: Expand beyond 3 bands"""
    band_mean = [[] for i in range(num_bands)]
    band_stdev = [[] for i in range(num_bands)]
    for i, file in enumerate(all_image_filepaths):
        current_img = misc.imread(file)
        
        for band in range(num_bands):
            band_mean[band].append(np.mean(current_img[:,:,band]))
            band_stdev[band].append(np.std(current_img[:,:,band]))
                
    for i,_ in enumerate(band_mean):
        band_mean[i] = np.mean(band_mean[i])
        band_stdev[i] = np.mean(band_stdev[i])
        
    return {'means': band_mean, 'sdevs':band_stdev}

def get_images(root_filepath, sort=True):
    """For a given path, returns a (sorted) list containing all
    files."""
    image_paths = []
    for folder, _, imgs in os.walk(root_filepath):
        for image_path in imgs:
            image_paths.append(os.path.join(folder, image_path))
    if sort is True:
        image_paths = sorted(image_paths)
    return image_paths

def keep_mixed_class_labels(img_paths, lbl_paths):
    """For any combination of image and label paths with
    identical filenames, Keeps only the labels that contain >1 class"""
    corrected_img_paths = []
    corrected_lbl_paths = []

    for i, label in enumerate(lbl_paths):
        lbl = np.array(Image.open(label))
        if np.max(lbl) != np.min(lbl):
            corrected_img_paths.append(img_paths[i])
            corrected_lbl_paths.append(lbl_paths[i])
    return corrected_img_paths, corrected_lbl_paths

def sync_img_and_lbls(root_dir, src_data_root, img_path, lbl_path):
    """Hotfix for accidentally deleted data.
    For any combination of image and label paths with
    identical filenames, copies files from source data path
    to ensure synchronous imgs & labels
    TODO: fix repetitive code, add conditional"""
    all_imgs = get_images(root_dir+img_path)
    all_lbls = get_images(root_dir+lbl_path)
    
    for file_path in all_imgs:
        file = os.path.basename(file_path)
        missing_lbl = root_dir + src_data_root + "labels/" + file
        copyfile(missing_lbl, root_dir + lbl_path + '/' + file)
    
    for file_path in all_lbls:
        file = os.path.basename(file_path)
        missing_img = root_dir + src_data_root + "tiles/" + file
        copyfile(missing_img, root_dir + img_path + '/' + file)

def replace_imgs_with_thumbnails(root_dir, width=400):
    for folder, _, imgs in os.walk(root_dir):
        for image_path in imgs:
            if not '.pickle' in image_path:
                # Credit: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
                full_img_path = os.path.join(folder, image_path)
                
                try:
                    img = Image.open(full_img_path)
                except Exception as e:
                    print(Exception)

                wpercent = (width/float(img.size[0]))
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((width,hsize), Image.ANTIALIAS)
                img.save(os.path.join(folder, image_path))

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