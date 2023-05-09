'''
    Dataset_CSV_SEM_SEG: Pytorch dataset class for semantic segmentation
        it can be used for both binary classification and multi-class classification with loss function of
          BCEWithLogitsLoss and CrossEntropyLoss, respectively.

'''

import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision.transforms import ToTensor
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
# from albumentations.pytorch.transforms import ToTensorV2



class Dataset_SEM_SEG(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False, mask_threshold=100):
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            assert Path(csv_file).exists(), f'csv file {csv_file} does not exists'
            self.csv_file = csv_file
            self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape

        list_transform = []  # A.NoOp
        if transform is not None:
            list_transform.extend(transform.transforms)
        if self.image_shape is not None:  # and (image.shape[:2] != self.image_shape[:2]):  # randomcrop can change the image size
            list_transform.append(A.Resize(height=self.image_shape[0], width=self.image_shape[1]))
        self.transform = A.Compose(
            list_transform,
        )
        self.test_mode = test_mode
        self.mask_threshold = mask_threshold

    def __getitem__(self, index):
        file_img = self.df.iloc[index][0]
        assert Path(file_img).exists(), f'image file {file_img} does not exists'
        image = cv2.imread(file_img)
        assert image is not None, f'{file_img} error.'
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.test_mode:
            file_mask = self.df.iloc[index][1]
            assert Path(file_mask).exists(), f'image mask file {file_mask} does not exists'
            mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, f'{file_mask} error.'

            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

            image = ToTensor()(image)  # convert numpy array to pytorch tensor and normalize to (0,1) using ToTensorV2 instead.

            if self.mask_threshold is not None:
                # pytorch (N, C, H, W), for get item, (C,H,W)
                mask_result = np.zeros((mask.shape[0], mask.shape[1]), np.uint)
                for i, mask_threshold in enumerate(self.mask_threshold):    # multi-class
                    if isinstance(mask_threshold, tuple):
                        tmp_mask = np.logical_and(mask > mask_threshold[0], mask <= mask_threshold[1]).astype(np.uint)
                        tmp_mask *= (i+1)
                    else:
                        _, tmp_mask = cv2.threshold(mask, mask_threshold, 1, cv2.THRESH_BINARY)  # mask_threshold:127, mask value : 0 or 1
                        tmp_mask *= (i+1)
                    mask_result += tmp_mask
            else:
                _, mask_result = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            mask_result = np.expand_dims(mask_result, axis=0)  # (H,W) -> (C,H,W) one channel output for binary and multi-class classification
            mask_result = torch.from_numpy(mask_result.astype(int))

            return image, mask_result
        else:
            image = self.transform(image=image)['image']
            image = ToTensor()(image)

            return image

    def __len__(self):
        return len(self.df)




def get_dataset(dataset, batch_size=32, num_workers=8, pin_memory=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    list_images = []
    list_labels = []

    for batch_idx, inputs in enumerate(dataloader):
        if isinstance(inputs, list):  # both images and masks
            images, labels = inputs

            images = images.cpu().numpy().astype(np.float16)  # B, C, H, W
            labels = labels.cpu().numpy().astype(np.float16)

            list_images.append(images)
            list_labels.append(labels)
        else:
            images = inputs.cpu().numpy().astype(np.float16)  # B, C, H, W
            list_images.append(images)


    if isinstance(inputs, list):
        images = np.concatenate(list_images, axis=0)
        labels = np.concatenate(list_labels, axis=0)

        return images, labels
    else:
        images = np.concatenate(list_images, axis=0)
        return images