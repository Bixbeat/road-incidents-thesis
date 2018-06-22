from os import path, walk
import numpy as np
import sys

import matplotlib.pyplot as plt
from scipy import misc
import time

from data_management import exif_functions
from data_management import image_manipulations as i_manips
from data_management.data_utils import ImgDatabaseHandler      

class ImageCleaner():
    def __init__(self, db_root, target_table):
        self.db_handler = ImgDatabaseHandler(db_root)
        self.target_table = target_table
        self.previous_img_path = None
        self.previous_geo = None
        self.previous_time = None        
        self.current_index = 0
        
    def skip_to_folder(self, all_folders, folder_name):
        index = all_folders.index(folder_name)
        folders_after_split = all_folders[index:]
        return(folders_after_split)
            

    def clean_images(self, analysis_folder, root_dir, target_class, skip_to_folder_name=None):
        self.db_handler.create_img_table(self.target_table)
        if skip_to_folder_name is not None:
            all_folders = [x[0] for x in walk(analysis_folder)]
            full_skip_folder = path.join(analysis_folder,skip_to_folder_name)
            folders_after_index = self.skip_to_folder(all_folders, full_skip_folder)
        print(f'''Determine whether image is of class {target_class}: 
                1 or empty is true, 0 is false, q to quit, sp to save previous, rp to remove previous, index to skip to an index''')
        for root, _, files in walk(analysis_folder):
            if skip_to_folder_name:
                if not root in folders_after_index:
                    continue
            print(f"\n\n\n\n\nNow in folder {root}\n\n\n\n\n")
            time.sleep(0.5) # Too easy to miss folder switches otherwise
            while self.current_index < len(files):
                if self.current_index < 0 or self.current_index > len(files):
                    self._set_index()
                    self.current_index += 1
                    continue
                img = files[self.current_index]
                img_path = path.join(root,img)
                if i_manips.is_image(img_path):
                    plt.figure(figsize = (8,8))
                    image = misc.imread(img_path) # Scikit because plotting PIL images doesn't work with Spyder QTConsole
                    plt.imshow(image, aspect='auto')
                    plt.show(block=False) # To force image render while user input is also in the pipeline
                    print(f'Index {self.current_index}: {img_path}')
                    response = str(input(f'Is this image representative of class {target_class}?: ')).lower()
                    self._handle_response(response, target_class, img_path, root_dir)
                self.current_index += 1
            self.current_index = 0
    
    def _handle_response(self, response, img_class, img_path, root_dir):
        time = -9999
        geo = ['','']
        path_without_root = img_path.split(root_dir)[1]
        if response in ['', '1', '2']:
            if response == '2': # Save image with different class name
                img_class = str(input(f'Which alternative image class is this image?: '))            
            
            img_exif = exif_functions.get_exif_if_exists(img_path)
            if img_exif:
                exif_with_geo = exif_functions.decode_geo(img_exif)
                if 'DateTimeOriginal' in img_exif.keys():
                    time = img_exif['DateTimeOriginal']
                if 'GPSInfo' in img_exif.keys():
                    geo = ['yes', 'yes'] # To implement later
            
            self.db_handler.store_image_details(self.target_table, img_class, path_without_root, geo, time)            
            
        elif response == 'sp':
            self.db_handler.store_image_details(self.target_table, img_class, self.previous_img_path, self.previous_geo, self.previous_time)
            response = str(input(f'Is this image representative of class {img_class}?: '))
            self._handle_response(response, img_class, img_path, root_dir)
        elif response == 'rp':
            self.db_handler.remove_record(self.target_table, self.previous_img_path)
            response = str(input(f'Is this image representative of class {img_class}?: '))
            self._handle_response(response, img_class, img_path, root_dir)
        elif response == '0':
            pass             
        elif response == 'index':
            self._set_index()
        elif response == 'q':
            sys.exit()
        else:
            print(f'''Determine whether image is of class {img_class}: 
                1 or empty for true, 0 for false, q to quit, sp to save previous, rp to remove previous''')            
            response = str(input(f'Is this image representative of class {img_class}?: '))
            self._handle_response(response, img_class, img_path, root_dir)
            
        self.previous_img_path = path_without_root     
        self.previous_geo = geo
        self.previous_time = time
            
    def _set_index(self):
        index = None
        while not type(index) == int:
            try:
                index = int(input(f'Type an index to skip to: '))
            except ValueError:
                print("Not an integer")
        self.current_index = index-1 #-1 to offset iteration increment        
            