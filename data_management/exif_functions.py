from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_if_exists(image):
    img = Image.open(image)
    if img.format in ['JPEG', 'TIFF']:
        exif = img._getexif()
        decoded_exif = decode_tags(exif)
        return decoded_exif

def decode_tags(exif):
    tagged_exif = {}
    if exif:
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
