#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""

import requests
import time
from os import path

from .utils import APICaller

class FlickrCaller(APICaller):
    def __init__(self, source, rest_url, api_key, data_root, returns_per_req):
        super().__init__(source, rest_url, api_key, data_root, returns_per_req)

    def download_tagged_images(self, tags, page, search_grouping):
        offset = self._assert_offset(page, self.returns_per_req)
        response = self.search_images(tags, page)
        photos = response.json()['photos']['photo']

        out_dir = self._construct_output_dir(search_grouping, tags)
        self._create_dir_if_not_exist(out_dir)     

        response_pickle = out_dir + f'/{tags}_{offset}.pickle'
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

    def search_images(self, tags, page=1):
        search_url = self._create_method_url('flickr.photos.search')
        params = {  'api_key':self.key,
                    'tags':tags,
                    'tag_mode':'all',
                    'page':str(page),
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