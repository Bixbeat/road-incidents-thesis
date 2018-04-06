#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""

from lib import scraper
from lib.scraper import GoogleCaller, FlickrCaller, BingCaller

def submit_query(api_caller, query, search_grouping, page):
    """Submit a query to the specified target.

    Args:
        apicaller(APICaller): API class of the target.
        query (string): Image search query to search for.
        search_grouping (string): Folder grouping for search results.
        page (int): The page index to start from.
    """
    api_caller.download_images(query, search_grouping = search_grouping, page = page)

if __name__ == '__main__':

## Build queries to automatically feed to APIs - Will result in many erratic searches, use as inspiration
    # combinations = scraper.get_query_combinations("street","snow")
    # extra_terms = ["city"] # Or add a list of synonyms, but this will create very many combinations
    # combinations = scraper.add_term_to_combinations(combinations, extra_terms)

    road_types = [['road'],['highway'],['street'],['route']]
    # landcovers = ['forest','countryside','city','mountain']
    snow_types = ['snow','blizzard']
    combinations = scraper.add_term_to_combinations(road_types, snow_types)

    # Define search parameters
    DATA_ROOT = '/media/alex/A4A034E0A034BB1E/incidents-thesis/data'
    search_grouping = "snowy_road"    

## Bing
    BING_API_KEY = u'' # From https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference 
    bing = BingCaller(BING_API_KEY, DATA_ROOT, returns_per_req = 100)

## Google
    GOOGLE_API_KEY = u'' # From https://console.developers.google.com
    CUSTOM_ENGINE = u'' # Create a custom search engine at https://cse.google.com
    google = GoogleCaller(GOOGLE_API_KEY, DATA_ROOT, returns_per_req = 10, cx = CUSTOM_ENGINE)

## Flickr
    FLICKR_API_KEY = u'' # From https://www.flickr.com/services/apps/ 
    flickr = FlickrCaller(FLICKR_API_KEY, DATA_ROOT, returns_per_req = 100)

## Querying
    for combination in combinations:
        if all(v is not None for v in [bing.error_code, google.error_code, flickr.error_code]):
            print("Errors exist in all API callers, cancelling search")
            break

        query = f"{combination[0]} {combination[1]}"
        submit_query(bing, query, search_grouping, page = 0)
        submit_query(flickr, query, search_grouping, page = 0)

        for i in range(10): # 10 imgs per call, max index is 100
            submit_query(google, query, search_grouping, page = i)