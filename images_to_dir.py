import random

from data_management import data_utils
from data_management.image_manipulations import image_to_dataloader_folders

if __name__ == '__main__':
    random.seed(1)
    incident_classes = ['snow', 'flooding', 'crash', 'animals', 'landslide', 'treefall', 'fire', 'collapse']
    root_dir = '/media/alex/A4A034E0A034BB1E/incidents-thesis'
    database = '/media/alex/A4A034E0A034BB1E/incidents-thesis/correctimgs'
    image_table_name = 'incidents'
    split_field_name = 'splits'
    split_probabilities = {'train': 70, 'val':20, 'test': 10}
    
    target_dir = '/media/alex/A4A034E0A034BB1E/incidents-thesis/incidents-cleaned'

    # handler.add_field('incidents', 'split') # Only once

    handler = data_utils.ImgDatabaseHandler(database)
    data_utils.create_dataloader_folders(root_dir, 'incidents-cleaned-small', incident_classes)

    handler.calculate_splits(image_table_name, split_field_name, split_probabilities)

    records = handler.get_all_records(image_table_name)
    
    for record in records:
        # dataloader_root, img_class, img_split, img_path
        image_to_dataloader_folders(target_dir, record[4], record[5], record[0], output_img_width=100, equal_aspect=True)
