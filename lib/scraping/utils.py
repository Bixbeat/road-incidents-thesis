import io
import os.path
import urllib
import pickle
from PIL import Image

class APICaller():
    def __init__(self, source, rest_url, api_key, data_root, returns_per_req):
        self.rest_url = rest_url
        self.source = source
        self.key = api_key
        self.data_root = data_root
        self.returns_per_req = returns_per_req # Max number of returns allowed per call

    def _save_image_file(self, image_bytes, path):
        with io.BytesIO(image_bytes.content) as f:
            f.seek(0)
            with Image.open(f) as img:
                img.save(path)

    def _create_dir_if_not_exist(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _construct_output_dir(self, search_grouping, tags):
        return(os.path.join(self.data_root, search_grouping, self.source, tags))

    def _store_response(self, response, pickle_file):
        with open(pickle_file, 'wb') as p:
            pickle.dump(response, p, protocol=pickle.HIGHEST_PROTOCOL)

    def _assert_offset(self, page, imgs_per_req):
        assert(page >= 0)
        return(page * imgs_per_req)        