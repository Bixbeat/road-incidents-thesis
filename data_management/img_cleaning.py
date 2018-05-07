from os import path, walk
import numpy as np
import sys

import matplotlib.pyplot as plt
from scipy import misc
import sqlite3

from data_management import exif_functions

class ImgDatabaseHandler():
    def __init__(self, db_root):
        self.db_root = None
        self.db = sqlite3.connect(db_root)
        self.cursor = self.db.cursor()        

    def _create_img_table(self, img_class):
        try:
            self.db.cursor.execute(f'''CREATE TABLE IF NOT EXISTS
                                    {img_class}(img_id VARCHAR PRIMARY KEY, lat REAL, long REAL, datetime VARCHAR)''')
            self.db.cursor.commit()
        except Exception as e:
            self.db.rollback()
            raise e

    def store_image_details(self, image_class, img_path, location, time):
        try:
            self.db.execute(f'INSERT INTO ? VALUES (?,?,?,?)', image_class, img_path, location[0], location[1], time)
        except sqlite3.IntegrityError:
            print('Record already exists')

class ImageCleaner():
    def __init__(self, db_root):
        self.db_handler = ImgDatabaseHandler(db_root)

    def clean_images(self, data_root, target_class):
        print(f'Determine whether image is of class {target_class} - 1 or empty is true, 0 is false, q for quit')
        for root, _, files in walk(data_root):
            for img in files:
                img_path = path.join(root,img)
                image = misc.imread(img_path) # Scikit because plotting PIL images doesn't work with Spyder QTConsole
                plt.imshow(image, aspect='auto')
                plt.show(block=False) # To force image render while user input is also in the pipeline
                
                response = str(input(f'Is this image representative of class {target_class}?: '))
                self._handle_response(response, target_class, img_path)

    def _handle_response(self, response, img_class, img_path):
        time = -9999
        geo = ['','']
        if response in ['', '1', '0', 'q']:
            if response in ['', '1']:
                img_exif = exif_functions.get_exif_if_exists(img_path)
                if img_exif:
                    exif_with_geo = exif_functions.decode_geo(img_exif)
                    if 'DateTimeOriginal' in img_exif.keys():
                        time = img_exif['DateTimeOriginal']
                    if 'GPSInfo' in img_exif.keys():
                        geo = ['yes', 'yes']
                        # To implement later

                self.db_handler.store_image_details(img_class, img_path, geo, time)
            elif response == '0':
                pass

            elif response == 'q':
                sys.exit()
        else:
            response = str(input(f'Is this image representative of class {img_class}?: '))
            self._handle_response(response, img_class, img_path)