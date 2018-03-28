#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:04:33 2018

@author: alex
"""
from thesaurusScraper import thesaurus as thes

query_term = thes.Word('street')
print(query_term.synonyms())