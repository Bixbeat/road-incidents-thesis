#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""
from lib.scraping import bing, google, flickr

if __name__ == '__main__':

## Bing
    # BING_API_URL = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search' 
    # DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
    # API_KEY = u'1386ef2c44454ad887c90db85ea90db7'
    # query = "snow highway"
    # search_grouping = "snowy_road"

    # searcher = bing.BingCaller( source = 'bing',
    #                             rest_url = BING_API_URL,
    #                             api_key = API_KEY,
    #                             data_root = DATA_ROOT,
    #                             returns_per_req = 150
    #                         )
    # for i in range(2):
    #     searcher.download_images(query, search_grouping = search_grouping, page = i)

## Google
    DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
    API_KEY = u'AIzaSyB5JGJMLDNYtawsP_8CPlX2c2ZhUiK-1CM' # From https://console.developers.google.com
    GOOGLE_API_URL = 'https://www.googleapis.com/customsearch/v1'
    CUSTOM_ENGINE = '013675800614641398741:wwg9y3xxkj0' # Create a custom search engine at https://cse.google.com

    search_grouping = "snowy_road"
    page = 0
    query = 'snow highway'
    size = 'medium'
    
    searcher = google.GoogleCaller(source='google',
                            rest_url = GOOGLE_API_URL,
                            api_key = API_KEY,
                            data_root = DATA_ROOT,
                            returns_per_req = 10,
                            cx = CUSTOM_ENGINE)
    for i in range(10):
        searcher.download_images(query, search_grouping = search_grouping, page = i, size = size)

# ## Flickr
#     DATA_ROOT = '/home/alex/Documents/Scripts/road-incidents-thesis/data/'
#     API_KEY = u'fa47986409159da74c70da5a438091af'
#     FLICKR_API_URL = 'https://api.flickr.com/services/rest/?'

#     search_grouping = "snowy_road"
#     page = 0
    
#     searcher = flickr.FlickrCaller(  source='flickr',
#                                 rest_url = FLICKR_API_URL,
#                                 api_key = API_KEY,
#                                 data_root = DATA_ROOT,
#                                 returns_per_req = 100)

#     searcher.download_tagged_images(tags = 'snow+road', search_grouping = search_grouping, page = page)