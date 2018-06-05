import random

from data_management import data_utils
from data_management.image_manipulations import image_to_dataloader_folders

if __name__ == '__main__':
    random.seed(1)
    root_dir = '/media/alex/A4A034E0A034BB1E/incidents-thesis'
    output_dir = '/media/alex/A4A034E0A034BB1E/incidents-thesis/final_data/incidents_cleaned'

    database = '/media/alex/A4A034E0A034BB1E/incidents-thesis/correctimgs'
    image_table_name = 'incidents'
    split_field_name = 'split'
    split_probabilities = {'train': 70, 'val':20, 'test': 10}

    handler = data_utils.ImgDatabaseHandler(database)
    try:
        handler.add_field(image_table_name, split_field_name) # Only once
    except Exception as e:
        print(e)
    
    handler.calculate_splits(image_table_name, split_field_name, split_probabilities)
    records = handler.get_all_records(image_table_name)

    all_recorded_classes = []
    [all_recorded_classes.append(str(record[4])) for record in records]
    unique_classes = set(all_recorded_classes)
    data_utils.create_dataloader_folders(root_dir, output_dir, unique_classes)
    
    for record in records:
        # dataloader_root, img_class, img_split, img_path
        image_to_dataloader_folders(output_dir, record[4], record[-1], record[0], output_img_width=500, equal_aspect=True)
