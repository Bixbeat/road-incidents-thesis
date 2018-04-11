import io
from io import BytesIO
import os.path
import pickle
import time
import requests 

import urllib
from PIL import Image

from data_collection.thesaurusScraper import thesaurus as th

class APICaller():
    """General API image searching wrapper.

    Base class containing common functionality between image search APIs.
    """    
    def __init__(self, source, rest_url, api_key, data_root, images_per_req):
        """
        Args:
            source (string): Description for saving purposes.
            rest_url (string): Target path for the API URL.
            api_key(string): The API key supplied for the API.
            data_root (string): The output path.
            images_per_req (int): The total amount of items to return per search.
        """        
        self.rest_url = rest_url
        self.source = source
        self.key = api_key
        self.data_root = data_root
        self.images_per_req = images_per_req # Max number of returns allowed per call

        self.error_code = None

    def _save_image_file(self, image_bytes, path):
        """Saves a bytes object to a specified target location.

        Args:
            image_bytes (byte): An image object.
            path (string): Output path for the image object.
        """          
        with io.BytesIO(image_bytes.content) as f:
            f.seek(0)
            with Image.open(f) as img:
                img.save(path)

    def _create_dir_if_not_exist(self, directory):
        """Creates a directory if the path does not yet exist.

        Args:
            directory (string): The directory to create.
        """          
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _construct_output_dir(self, search_grouping, query):
        """Creates a directory path for a search.

        Args:
            search_grouping (string): Folder grouping for search results.
            query (string): Image search query to search for.

        Returns:
            The constructed output directory.
        """        
        return(os.path.join(self.data_root, search_grouping, self.source, query))

    def _store_response(self, response, pickle_file):
        """Pickles a response file for later processing.

        Args:
            response (dict): Response object retrieved from API.
            pickle_file: File path of the pickle to save.
        """                
        with open(pickle_file, 'wb') as p:
            pickle.dump(response, p, protocol=pickle.HIGHEST_PROTOCOL)

    def _assert_offset(self, page, imgs_per_req):
        """Determines the starting index based on the page and the number of requests

        Args:
            page (int): The page index to start from.
            images_per_req (int): The total amount of items to return per search.

        Returns:
            The offset based on the page number and images per request.
        """        
        assert(page >= 0)
        return(page * imgs_per_req)      

    def _check_status_code(self, status_code):
        """Check if the last run resulted in an error code from the API.

        Args:
            status_code (int): The status code returned by the API call.
        """        
        if status_code != 200:
            self.error_code = status_code
            print(f'aborting further execution, error code {status_code} received for {self.source} caller')
    
    def _check_if_key_in_dict(self, key, results):
        """Check whether a given key exists in a dictionary.

        Args:
            key (string): The key to search for.
            results (dict): The dictionary containing the API response.
        
        Returns:
            False if the key does not exist
        """           
        if not key in results.keys():
            print(f"Response dict does not contain key {key}")
            return False

class GoogleCaller(APICaller):
    """Subclass for calling Google API calls & handling response.

    See the following link for a more extensive overview of the set-up:
    https://stackoverflow.com/questions/34035422/google-image-search-says-api-no-longer-available
    """
    def __init__(self, api_key, data_root, returns_per_req, cx):
        super().__init__('google',
                         'https://www.googleapis.com/customsearch/v1',
                         api_key,
                         data_root,
                         returns_per_req)
        self.cx = cx
        self.img_size = 'medium'

    def download_images(self, query, page, search_grouping):
        if self.error_code:
            return 0 # Prevent repeated API calls when error is received

        offset = self._assert_offset(page, self.images_per_req)
        params  = { 'key': self.key,
                    'gl':'uk',
                    'googlehost':'google.uk',
                    'cx':self.cx,
                    'q':query,
                    'searchType':'image',
                    'imageSize':self.img_size,
                    'filter':'1',
                    'imgType':'photo',
                    'num':self.images_per_req,
                }
        if offset > 0:
            params['start'] = offset # Offset must be between 1 and 90

        response = requests.get(self.rest_url, params=params)
        self._check_status_code(response.status_code)

        search_results = response.json()
        
        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)

        response_pickle = out_dir + f'/{query}_{self.img_size}_{offset}.pickle'
        self._store_response(response, response_pickle)

        if self._check_if_key_in_dict('items',search_results) == False:
            return None
        for i, search_result in enumerate(search_results['items']):
            try:
                image_bytes = requests.get(search_result['link'], timeout=10)
            except Exception as e:
                print(f"Unreachable URL: {search_result['link']}\n{str(e)}\n")

            image_path = out_dir + f'/{self.img_size}_{offset+i+1}.png'

            try:
                self._save_image_file(image_bytes, image_path)
            except Exception as e:
                print(f"Unsaveable image: {search_result['link']}\n{str(e)}\n")

class BingCaller(APICaller):
    """Subclass for calling Google API calls & handling response.

    See the following link for the API reference:
    https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference
    """    
    def __init__(self, api_key, data_root, returns_per_req):
        super().__init__('bing',
                         'https://api.cognitive.microsoft.com/bing/v7.0/images/search',
                         api_key,
                         data_root,
                         returns_per_req)

    def download_images(self, query, page, search_grouping):
        if self.error_code:
            return None # Prevent repeated API calls when error is received        

        offset = self._assert_offset(page, self.images_per_req)
        headers = {'Ocp-Apim-Subscription-Key' : self.key}
        params  = { 'q': query,
                    # 'license': 'shareCommercially',
                    'imageType': 'photo',
                    'count':self.images_per_req,
                    'offset':offset
                }

        response = requests.get(self.rest_url, headers=headers, params=params)
        self._check_status_code(response.status_code)

        if self.error_code:
            return None # Prevent repeated API calls when error is received

        search_results = response.json()        
        
        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)

        response_pickle = out_dir + f'/{query}_{offset}.pickle'
        self._store_response(response, response_pickle)

        if self._check_if_key_in_dict('value',search_results) == False: return 0
        for search_result in search_results['value']:
            image_id = search_result['imageId']

            try:
                image_bytes = requests.get(search_result['contentUrl'], timeout=10)
            except Exception as e:
                print(f"Unreachable URL: {search_result['contentUrl']}\n{str(e)}\n")

            image_path = out_dir + f'/{image_id}.png'

            try:
                self._save_image_file(image_bytes, image_path)
            except Exception as e:
                print(f"Unsaveable image: {search_result['contentUrl']}\n{str(e)}\n")

class FlickrCaller(APICaller):
    """Subclass for calling Flickr API calls & handling response.

    Uses only the photo search API call and the image ID lookup. More info on params here:
    https://www.flickr.com/services/api/flickr.photos.search.htm
    """     
    def __init__(self, api_key, data_root, returns_per_req):
        super().__init__('flickr',
                         'https://api.flickr.com/services/rest/?',
                         api_key,
                         data_root,
                         returns_per_req)

    def download_images(self, query, page, search_grouping):
        if self.error_code:
            return None # Prevent repeated API calls when error is received        

        offset = self._assert_offset(page, self.images_per_req)
        response = self.search_images(query, page)
        self._check_status_code(response.status_code)
        

        search_results = response.json()        

        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)     

        response_pickle = out_dir + f'/{query}_{offset}.pickle'
        self._store_response(response, response_pickle)
        
        if self._check_if_key_in_dict('photos',search_results) == False:
            return None

        photos = search_results['photos']['photo']            
        for _,photo in enumerate(photos):
            image_id = photo['id']
            sizes_response  = self.get_image_sizes(image_id)

            if self._check_if_key_in_dict('sizes',sizes_response.json()) != False:
                img_sizes = sizes_response.json()['sizes']
            
                if not img_sizes['candownload'] == 0:
                    highest_res_url = self._get_image_url(img_sizes, resolution = 7)

                    try:
                        image_bytes = image_bytes = requests.get(highest_res_url, timeout=10)
                    except Exception as e:
                        print(f"Unreachable URL: {highest_res_url}\n{str(e)}\n")

                    image_path = out_dir + f'/{image_id}.png'

                    try:
                        self._save_image_file(image_bytes, image_path)
                    except Exception as e:
                        print(f"Unsaveable image: {image_bytes}\n{str(e)}\n")
                                
                time.sleep(0.2) # Restricting API call frequency to be a good citizen
            else:
                print("Empty sizes dictionary, skipping.")

    def search_images(self, query, page):
        """Queries the Flickr API.

        Args:
            query (string): Image search query to search for.
            page (int): The page index to start from.
        
        Returns:
            The API response.

        TODO: Generalize to other classes with params dict as input to supply.
        """        
        search_url = self._create_method_url('flickr.photos.search')
        params = {  'api_key':self.key,
                    'text':query,
                    'tag_mode':'all',
                    'per_page':self.images_per_req,
                    'page':str(page),
                    'sort':'relevance',
                    'media':'photos',
                    'format':'json',
                    'nojsoncallback':1,
                }
        
        response = requests.get(search_url, params = params)
        return response

    def get_image_sizes(self, image_id):
        """Queries the Flickr API for image formats of a given image ID.

        Args:
            image_id (string): The unique image identifier found in the API response.
        
        Returns:
            A dict of all image formats.
        """        
        size_url = self._create_method_url('flickr.photos.getSizes')
        params = {  'api_key':self.key,
                    'photo_id':image_id,
                    'format':'json',
                    'nojsoncallback':1
                }        
        response = requests.get(size_url, params = params)
        return response        

    def _get_image_url(self, img_sizes, resolution = 7):
        """For a given stack of URLs, gets the specified resolution image.

        Does not check the actual resolution but instead uses the highest node number in the list,
        which by default contains the highest resolution. If the resolution is greater than the specified
        resolution, returns the specified resolution.

        Args:
            method (string): The method that the Flickr API has to execute.
            resolution (int): The resolution specified by the node. Lower values are lower resolutions.
        
        Returns:
            The URL of the specified resolution image, or lower res if not applicable.

        TODO: Find a more elegant solution.
        """
        image_node = [i for i,_ in enumerate(img_sizes['size'])][-1] #Get highest
        if image_node > resolution:
            image_node = resolution # Reduce the 
        highest_res_url = img_sizes['size'][image_node]['source']

        return(highest_res_url)

    def _create_method_url(self, method):
        """Appends the Flickr method to the Flickr API url.

        Args:
            method (string): The method that the Flickr API has to execute.
        
        Returns:
            The Flickr method URL.
        """           
        return f"{self.rest_url}method={method}"                


def get_query_combinations(first_term, second_term):
    """Search Thesaurus for synonyms, then create a list of two of all possible combinations.

    Args:
        first_term (string): First keyword to search for.
        second_term (string): Second keyword to search for.
    
    Returns:
        A list of all possible synonym-combinations of the two terms.
    """        
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
    """For a given list with combinations, add an extra term to every list.

    The input for terms can contain 1 or more word in the list. It will then create a new list
    for every new possible combination.

    Args:
        combinations (list of lists): A list of N sublists containing search terms.
        terms (list of strings): A list of N words to create new combination search terms with.
    
    Returns:
        A new combination list containing the new set of possible combinations.
    """        
    combos = []
    for term in terms:
        for combo in combinations:
            combos.append(combo+[term])
    return combos