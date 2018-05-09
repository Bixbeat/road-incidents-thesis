import string
import random

import os
from PIL import Image

def create_dir_if_not_exist(directory):
    """Creates a directory if the path does not yet exist.

    Args:
        directory (string): The directory to create.
    """          
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_random_filename(length=10):
    # https://www.pythoncentral.io/python-snippets-how-to-generate-random-string/
    allchar = string.ascii_letters + string.digits
    return("".join(random.choice(allchar) for x in range(length)))