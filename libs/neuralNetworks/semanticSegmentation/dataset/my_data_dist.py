'''

'''

from pathlib import Path
from libs.neuralNetworks.semanticSegmentation.dataset.my_dataset import Dataset_SEM_SEG
import numpy as np
import cv2



def get_dataset_distribution(csv_file, mask_threshold=127):
    dataset1 = Dataset_SEM_SEG(csv_file=csv_file, mask_threshold=mask_threshold)

    gt_positives, gt_negatives = 0, 0
    for index, (image, mask) in enumerate(dataset1):
        mask = mask.numpy()  # (C=1,H,W)
        gt_negatives += np.sum(mask == 0)
        gt_positives += np.sum(mask == 1)

    print(gt_positives, gt_negatives, gt_negatives / gt_positives)


def get_path_distribution(path1: Path, mask_threshold=127):

    gt_positives, gt_negatives = 0, 0

    for item1 in path1.rglob('*.*'):
        if item1.suffix not in ['.png', '.jpg', '.jpeg']:
            continue
        mask_file = item1.parent / f'{item1.stem}_mask.jpg'
        if not mask_file.exists():
            continue

        image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        gt_negatives += np.sum(image < mask_threshold)
        gt_positives += np.sum(image > mask_threshold)

    print(gt_positives, gt_negatives, gt_negatives / gt_positives)



if __name__ == '__main__':

    for task_type in ['cyto_clumps', 'cyto_ins', 'nuc_clumps', 'nuc_ins']:

        path_csv = Path(__file__).resolve().parents[4] / 'datafiles' / f'{task_type}' / 'train_valid.csv'
        get_dataset_distribution(path_csv)

    '''
    3.598541957809802
    3.9062166406712957
    125.04773719539169
    125.04773719539169
    '''
    print('OK')