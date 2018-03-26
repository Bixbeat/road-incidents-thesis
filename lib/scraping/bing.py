#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:13:08 2018

@author: alex
"""

from io import BytesIO
import os.path
import requests

from utils import APICaller

# Base class should be refactored & combined with the flickrsearcher class due to very high similarity
# Might do that if I have time down the line
class BingCaller(APICaller):
    def __init__(self, source, rest_url, api_key, data_root, returns_per_req):
        super().__init__(source, rest_url, api_key, data_root, returns_per_req)
    def download_images(self, query, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        headers = {'Ocp-Apim-Subscription-Key' : self.key}
        params  = { 'q': query,
                    'license': 'public',
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
            try: image_bytes = requests.get(search_result['contentUrl'], stream=True)
            except Exception as e:
                print(f"Unreachable URL: {search_result['contentUrl']}\n{str(e)}\n")
                break

            image_path = out_dir + f'/{image_id}.png'
            try: self._save_image_file(image_bytes, image_path)
            except Exception as e: print(f"Unsaveable image: {search_result['contentUrl']}\n{str(e)}\n")

if __name__ == '__main__':
    BING_API_URL = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search' 
    DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
    API_KEY = u''
    query = "Snow on road"
    search_grouping = "snowy_road"

    searcher = BingCaller(  source = 'bing',
                            rest_url = BING_API_URL,
                            api_key = API_KEY,
                            data_root = DATA_ROOT,
                            returns_per_req = 150
                        )
    searcher.download_images(query, page=0, search_grouping = search_grouping)
