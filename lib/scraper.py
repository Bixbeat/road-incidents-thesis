import io
from io import BytesIO
import os.path
import pickle
import time
import requests 

import urllib
from PIL import Image

from lib.thesaurusScraper import thesaurus as th

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

    def _construct_output_dir(self, search_grouping, query):
        return(os.path.join(self.data_root, search_grouping, self.source, query))

    def _store_response(self, response, pickle_file):
        with open(pickle_file, 'wb') as p:
            pickle.dump(response, p, protocol=pickle.HIGHEST_PROTOCOL)

    def _assert_offset(self, page, imgs_per_req):
        assert(page >= 0)
        return(page * imgs_per_req)      

class GoogleCaller(APICaller):
    # https://stackoverflow.com/questions/34035422/google-image-search-says-api-no-longer-available
    def __init__(self, api_key, data_root, returns_per_req, cx):
        super().__init__('google',
                         'https://www.googleapis.com/customsearch/v1',
                         api_key,
                         data_root,
                         returns_per_req)
        self.cx = cx
        self.img_size = 'medium'

    def download_images(self, query, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        params  = { 'key': self.key,
                    'gl':'uk',
                    'googlehost':'google.uk',
                    'cx':self.cx,
                    'q':query,
                    'searchType':'image',
                    'imageSize':self.img_size,
                    'filter':'1',
                    'imgType':'photo',
                    'num':self.returns_per_req,
                }
        if offset > 0: params['start'] = offset # Offset must be between 1 and 90

        response = requests.get(self.rest_url, params=params)
        search_results = response.json()
        
        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)

        response_pickle = out_dir + f'/{query}_{self.img_size}_{offset}.pickle'
        self._store_response(response, response_pickle)

        for i, search_result in enumerate(search_results['items']):
            try: image_bytes = requests.get(search_result['link'], timeout=10)
            except Exception as e: print(f"Unreachable URL: {search_result['link']}\n{str(e)}\n")

            image_path = out_dir + f'/{self.img_size}_{offset+i+1}.png'
            try: self._save_image_file(image_bytes, image_path)
            except Exception as e: print(f"Unsaveable image: {search_result['link']}\n{str(e)}\n")

class BingCaller(APICaller):
    def __init__(self, api_key, data_root, returns_per_req):
        super().__init__('bing',
                         'https://api.cognitive.microsoft.com/bing/v7.0/images/search',
                         api_key,
                         data_root,
                         returns_per_req)

    def download_images(self, query, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        headers = {'Ocp-Apim-Subscription-Key' : self.key}
        params  = { 'q': query,
                    # 'license': 'shareCommercially',
                    'imageType': 'photo',
                    'count':self.returns_per_req,
                    'offset':offset
                }

        response = requests.get(self.rest_url, headers=headers, params=params)
        search_results = response.json()
        
        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)

        response_pickle = out_dir + f'/{query}_{offset}.pickle'
        self._store_response(response, response_pickle)

        for search_result in search_results['value']:
            image_id = search_result['imageId']
            try: image_bytes = requests.get(search_result['contentUrl'], timeout=10)
            except Exception as e: print(f"Unreachable URL: {search_result['contentUrl']}\n{str(e)}\n")

            image_path = out_dir + f'/{image_id}.png'
            try: self._save_image_file(image_bytes, image_path)
            except Exception as e: print(f"Unsaveable image: {search_result['contentUrl']}\n{str(e)}\n")

class FlickrCaller(APICaller):
    def __init__(self, api_key, data_root, returns_per_req):
        super().__init__('flickr',
                         'https://api.flickr.com/services/rest/?',
                         api_key,
                         data_root,
                         returns_per_req)

    def download_tagged_images(self, query, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        response = self.search_images(query, page)
        photos = response.json()['photos']['photo']

        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)     

        response_pickle = out_dir + f'/{query}_{offset}.pickle'
        self._store_response(response, response_pickle)

        for i,_ in enumerate(photos):
            image_id = photos[i]['id']
            sizes_response  = self.get_image_sizes(image_id)
            img_sizes = sizes_response.json()['sizes']
        
            if not img_sizes['candownload'] == 0:
                highest_res_url = self._get_highest_resolution_img(img_sizes)             
                try: image_bytes = image_bytes = requests.get(highest_res_url, timeout=10)
                except Exception as e: print(f"Unreachable URL: {highest_res_url}\n{str(e)}\n")

                image_path = out_dir + f'/{image_id}.png'
                try: self._save_image_file(image_bytes, image_path)
                except Exception as e: print(f"Unsaveable image: {image_bytes}\n{str(e)}\n")
                            
            time.sleep(0.2) # Restricting API call frequency to be a good citizen

    def search_images(self, query, page=1):
        search_url = self._create_method_url('flickr.photos.search')
        params = {  'api_key':self.key,
                    'text':query,
                    'tag_mode':'all',
                    'page':str(page),
                    'sort':'relevance',
                    'media':'photos',
                    'format':'json',
                    'nojsoncallback':1,
                }
        
        response = requests.get(search_url, params = params)
        return response

    def get_image_sizes(self, image_id):
        size_url = self._create_method_url('flickr.photos.getSizes')
        params = {  'api_key':self.key,
                    'photo_id':image_id,
                    'format':'json',
                    'nojsoncallback':1
                }        
        response = requests.get(size_url, params = params)
        return response        

    def _get_highest_resolution_img(self, img_sizes):
        # There has got to be a better way to find the highest resolution..
        highest_res_node = [i for i,_ in enumerate(img_sizes['size'])][-1]
        highest_res_url = img_sizes['size'][highest_res_node]['source']

        return(highest_res_url)

    def _create_method_url(self, method):
        return f"{self.rest_url}method={method}"                


def get_query_combinations(first_term, second_term):
    all_combinations = []

    synonyms_1 = th.Word(first_term).synonyms()
    syn1 = [first_term]+synonyms_1 # pre-prending original search terms
    synonyms_2 = th.Word(second_term).synonyms()
    syn2 = [second_term]+synonyms_2

    for s1 in syn1:
        for s2 in syn2:
            all_combinations.append([s1, s2])
    
    return all_combinations

def add_term_to_combinations(combinations, terms):
    combos = []
    for term in terms:
        for combo in combinations:
            combos.append(combo+[term])
    return combos