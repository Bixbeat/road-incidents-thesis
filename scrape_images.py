#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""
from lib import scraper
from lib.scraper import GoogleCaller, FlickrCaller, BingCaller

def submit_query(api_caller, query, search_grouping, page):
    api_caller.download_images(query, search_grouping = search_grouping, page = 0)

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
    DATA_ROOT = '/media/alex/A4A034E0A034BB1E/incidents-thesis'
    search_grouping = "snowy_road"    

## Bing
    BING_API_KEY = u''
    bing = BingCaller(BING_API_KEY, DATA_ROOT, returns_per_req = 100)

## Google
    GOOGLE_API_KEY = u'-1CM' # From https://console.developers.google.com
    CUSTOM_ENGINE = '013675800614641398741:wwg9y3xxkj0' # Create a custom search engine at https://cse.google.com
    google = GoogleCaller(GOOGLE_API_KEY, DATA_ROOT, returns_per_req = 10, cx = CUSTOM_ENGINE)

## Flickr
    FLICKR_API_KEY = u''
    flickr = FlickrCaller(FLICKR_API_KEY, data_root = DATA_ROOT, returns_per_req = 100)

## Querying
    for combination in combinations:
        query = f"{combination[0]} {combination[1]}"                  
        for i in range(10):
            submit_query(google, query, search_grouping, page = i)    