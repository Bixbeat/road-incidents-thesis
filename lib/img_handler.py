import os.path

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
            
    def save_pil_image(image, path):
        pass
