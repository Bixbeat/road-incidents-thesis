import shutil
import urllib

url = 'https://farm5.staticflickr.com/4793/39830059435_e3c1cb2df0_o.jpg'
response = requests.get(url, stream=True)
with open('/home/alex/Documents/Scripts/road-incidents-thesis/lib/scraping/img.png', 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)