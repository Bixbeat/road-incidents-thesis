from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif(image):
    img = Image.open(image)
    exif = img._getexif()
    return exif

def decode_tags(exif):
    tagged_exif = {}
    for tag, value in exif.items():
        decoded_tag = TAGS.get(tag, tag)
        tagged_exif[decoded_tag] = value
    return tagged_exif

def decode_geo(exif_dict):
    if 'GPSInfo' in exif_dict.keys():

        if len(exif_dict['GPSInfo']) > 1: # Invalid tags may occur with len 1
            gps_data = {}

            for tag in exif_dict['GPSInfo']:
                sub_decoded = GPSTAGS.get(tag, tag)
                gps_data[sub_decoded] = exif_dict['GPSInfo'][tag]
            exif_dict['GPSInfo'] = gps_data
            
    return exif_dict
    