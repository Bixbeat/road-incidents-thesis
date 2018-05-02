from data_management import exif_functions

if __name__ == '__main__':
    coded_exif = exif_functions.get_exif('/home/alex/Documents/test.jpg')
    exif = exif_functions.decode_tags(coded_exif)
    geocoded_exif = exif_functions.decode_geo(exif)
    print(geocoded_exif)