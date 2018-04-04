#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""
from lib import scraper
from lib.scraper import GoogleCaller, FlickrCaller, BingCaller

if __name__ == '__main__':

## Build queries to automatically feed to APIs
    combinations = scraper.get_query_combinations("street","snow")
    extra_terms = ["city"] # Or add a list of synonyms, but this will create very many combinations
    combinations = scraper.add_term_to_combinations(combinations, extra_terms)

#    # Skip to index
#    combinations = combinations[23:]

### Bing
#    BING_API_URL = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search' 
#    DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
#    API_KEY = u''
#    search_grouping = "snowy_road"
#
#    searcher = bing.BingCaller( source = 'bing',
#                                rest_url = BING_API_URL,
#                                api_key = API_KEY,
#                                data_root = DATA_ROOT,
#                                returns_per_req = 50
#                            )
#    for combination in combinations:
#        query = f"{combination[0]} {combination[1]} {combination[2]}"
#        searcher.download_images(query, search_grouping = search_grouping, page = 0)

## Google
    # DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
    # API_KEY = u'' # From https://console.developers.google.com
    # GOOGLE_API_URL = 'https://www.googleapis.com/customsearch/v1'
    # CUSTOM_ENGINE = '013675800614641398741:wwg9y3xxkj0' # Create a custom search engine at https://cse.google.com

    # search_grouping = "snowy_road"
    # page = 0
    # query = 'snow city street'
    # size = 'medium'
    
    # searcher = google.GoogleCaller(source='google',
    #                         rest_url = GOOGLE_API_URL,
    #                         api_key = API_KEY,
    #                         data_root = DATA_ROOT,
    #                         returns_per_req = 10,
    #                         cx = CUSTOM_ENGINE)
    # for i in range(10):
    #     searcher.download_images(query, search_grouping = search_grouping, page = i, size = size)

## Flickr
    DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
    API_KEY = u''
    FLICKR_API_URL = 'https://api.flickr.com/services/rest/?'

    search_grouping = "snowy_road"
    page = 0

    searcher = FlickrCaller(source='flickr',
                            rest_url = FLICKR_API_URL,
                            api_key = API_KEY,
                            data_root = DATA_ROOT,
                            returns_per_req = 100)
    for combination in combinations:
        query = f"{combination[0]} {combination[1]}"
        searcher.download_tagged_images(query = query, search_grouping = search_grouping, page = page)