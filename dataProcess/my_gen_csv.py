'''

'''

from pathlib import Path
from libs.dataPreprocess.my_data import write_csv, split_dataset, write_list_csv


for task_type in ['cyto_clumps', 'cyto_ins', 'nuc_clumps', 'nuc_ins']:
    path_csv = Path(__file__).resolve().parent.parent / 'datafiles' /task_type
    path_csv.mkdir(parents=True, exist_ok=True)

    path_base = Path('/disk_data/data/carvical/Cx22/patches')

    path_images = path_base / 'training' / 'images'
    path_masks = path_base / 'training' / 'masks'
    write_csv(path_csv / 'train_valid.csv', path_images, path_masks, mask_ext=f'{task_type}.jpg')

    list_train_image, list_train_mask, list_valid_image, list_valid_mask = split_dataset(path_csv / 'train_valid.csv', valid_ratio=0.1)
    write_list_csv(path_csv / 'train.csv', list_train_image, list_train_mask)
    write_list_csv(path_csv / 'valid.csv', list_valid_image, list_valid_mask)

    path_images = path_base / 'test' / 'images'
    path_masks = path_base / 'test' / 'masks'
    write_csv(path_csv / 'test.csv', path_images, path_masks, mask_ext=f'{task_type}.jpg')


print('OK')

