import io
import os.path
import urllib
from PIL import Image

def write_img_from_url(image, path):
    with io.BytesIO(image.content) as f:
        with Image.open(f) as img:
            img.save(path)

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)