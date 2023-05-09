import os
import csv
import pandas as pd
import sklearn
from pathlib import Path



def write_csv(filename_csv, path_images: Path, path_masks: Path, mask_ext='cyto_clumps.jpg', field_columns=['images', 'masks']):
    if filename_csv.exists():
        os.remove(filename_csv)
    filename_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(field_columns)

        for file_img in path_images.rglob('*'):
            if not file_img.is_file():
                continue
            if file_img.suffix not in ['.jpg', '.png']:
                continue

            file_mask = path_masks / f'{file_img.stem}_{mask_ext}'
            if not file_mask.exists():
                continue
            csv_writer.writerow((str(file_img), str(file_mask)))



def split_dataset(filename_csv, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'masks']):
    
    df = pd.read_csv(filename_csv)
    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df) * (1 - valid_ratio))
        data_train = df[:split_num]
        list_train_image = data_train[field_columns[0]].tolist()
        list_train_mask = data_train[field_columns[1]].tolist()

        data_valid = df[split_num:]
        list_valid_image = data_valid[field_columns[0]].tolist()
        list_valid_mask = data_valid[field_columns[1]].tolist()

        return list_train_image, list_train_mask, list_valid_image, list_valid_mask
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        list_train_image = data_train[field_columns[0]].tolist()
        list_train_mask = data_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        list_valid_image = data_valid[field_columns[0]].tolist()
        list_valid_mask = data_valid[field_columns[1]].tolist()

        data_test = df[split_num_valid:]
        list_test_image = data_test[field_columns[0]].tolist()
        list_test_mask = data_test[field_columns[1]].tolist()

        return list_train_image, list_train_mask, \
               list_valid_image, list_valid_mask, list_test_image, list_test_mask




def write_list_csv(filename_csv, list_image_files, list_mask_files, field_columns=['images', 'masks']):
    if filename_csv.exists():
        os.remove(filename_csv)
    filename_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(field_columns)

        for image_file, mask_file in zip(list_image_files, list_mask_files):
            csv_writer.writerow((str(image_file), mask_file))


