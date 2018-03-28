#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:13:08 2018

@author: alex
https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-web-api-v7-reference
"""

from io import BytesIO
import os.path
import requests

from .utils import APICaller

# Base class should be refactored & combined with the flickrsearcher class due to very high similarity
# Might do that if I have time down the line
class BingCaller(APICaller):
    def __init__(self, source, rest_url, api_key, data_root, returns_per_req):
        super().__init__(source, rest_url, api_key, data_root, returns_per_req)
    def download_images(self, query, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        headers = {'Ocp-Apim-Subscription-Key' : self.key}
        params  = { 'q': query,
                    'license': 'shareCommercially',
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
