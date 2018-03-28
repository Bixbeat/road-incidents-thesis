#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
https://stackoverflow.com/questions/34035422/google-image-search-says-api-no-longer-available
"""
from io import BytesIO
import os.path
import requests

from .utils import APICaller

# Base class should be refactored & combined with the flickrsearcher class due to very high similarity
# Might do that if I have time down the line
class GoogleCaller(APICaller):
    def __init__(self, source, rest_url, api_key, data_root, returns_per_req, cx):
        super().__init__(source, rest_url, api_key, data_root, returns_per_req)
        self.cx = cx

    def download_images(self, query, page, search_grouping, size):
        offset = self._assert_offset(page, self.returns_per_req)
        # headers = {'key': self.key}
        params  = { 'key': self.key,
                    'gl':'uk',
                    'googlehost':'google.uk',
                    'cx':self.cx,
                    'q':query,
                    'searchType':'image',
                    'imageSize':size,
                    'filter':'1',
                    'imgType':'photo',
                    'num':self.returns_per_req,
                }
        if offset > 0: params['start'] = offset # Offset must be between 1 and 90

        response = requests.get(self.rest_url, params=params)
        search_results = response.json()
        
        out_dir = self._construct_output_dir(search_grouping, query)
        self._create_dir_if_not_exist(out_dir)

        response_pickle = out_dir + f'/{query}_{size}_{offset}.pickle'
        self._store_response(response, response_pickle)

        for i, search_result in enumerate(search_results['items']):
            try: image_bytes = requests.get(search_result['link'], timeout=10)
            except Exception as e: print(f"Unreachable URL: {search_result['link']}\n{str(e)}\n")

            image_path = out_dir + f'/{size}_{offset+i+1}.png'
            try: self._save_image_file(image_bytes, image_path)
            except Exception as e: print(f"Unsaveable image: {search_result['link']}\n{str(e)}\n")