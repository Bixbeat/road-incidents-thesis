import io
import os.path
import urllib
import pickle
from PIL import Image

def write_img_from_url(image, path):
    with io.BytesIO(image.content) as f:
        with Image.open(f) as img:
            img.save(path)

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def construct_output_dir(data_root, search_grouping, tags, source):
    return(os.path.join(data_root, search_grouping, tags))

def store_response(response, pickle_file):
    with open(pickle_file, 'wb') as p:
        pickle.dump(response, p, protocol=pickle.HIGHEST_PROTOCOL)

def assert_offset(page, imgs_per_req):
    assert(page >= 0)
    return(page * imgs_per_req)