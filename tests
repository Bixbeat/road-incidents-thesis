#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:13:08 2018

@author: alex
"""
from lib import img_handler

if __name__ == '__main__':
    # TODO: Replace to separate file
    # TODO: Sort out unittest environment
    # https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
    import unittest
    
    query_handler = img_handler.Handler()
    
    class TestStringMethods(unittest.TestCase):
            
        def test_write_query(self):
            query_handler.store_query("Snowy roads", 2)
    
    unittest.main()