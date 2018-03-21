import os.path
import shutil

class Handler():
    def __init__(self, data_folder='data'):
        self.root = data_folder
        self.query_log = 'queries'
        
    def store_query(self,query, id_num):
        query_path = os.path.join(self.root, 'log', self.query_log)
        if not os.path.isfile(query_path):mode = 'w'
        else: mode = 'a'
        
        with open(query_path, mode) as file:
            if mode == 'w': file.write("id_num,query\n")
            file.write(f"{id_num}, {self.query_log}\n")
            
    def save_pil_image(self, image, path):
        pass

def write_img_from_url(image, path):
    with open(path, 'wb') as out_file:
        shutil.copyfileobj(image.raw, out_file)
