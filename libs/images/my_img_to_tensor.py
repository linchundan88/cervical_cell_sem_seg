'''
img_to_tensor: input the filename, output a tenser that can be input to neural network models.
    because the parameter transform, it can be used in test time augmentation.
tensor_to_image: predicted results restore to images.
'''

import torch
import albumentations as A
from torchvision.transforms import ToTensor
import numpy as np
import cv2


def img_to_tensor(img_file, image_shape=None, transform=None):
    if isinstance(img_file, str):
        image = cv2.imread(img_file)
    else:
        image = img_file

    # if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
    #     image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

    list_transform = []
    # if transform is not None:
    #     list_transform.extend(transform.transforms)
    if transform is not None:
        list_transform.append(transform)
    elif image_shape is not None:
        list_transform.append(A.Resize(height=image_shape[0], width=image_shape[1]))
    else:
        list_transform.append(A.NoOp)

    transform1 = A.Compose(list_transform)
    image = transform1(image=image)['image']

    tensor = ToTensor()(image)
    tensor = torch.unsqueeze(tensor, dim=0)  # (C,H,W) -> (N,C,H,W)

    return tensor


def tensor_to_image(tensor1, img_file=None):
    array1 = tensor1.cpu().numpy()[0]  #(N,C,H,W) -> (C,H,W)
    array1 = np.transpose(array1, axes=(1, 2, 0))  # (C,H,W)
    array1 *= 255

    if img_file is not None:
        cv2.imwrite(img_file, array1)

    return array1



if __name__ == '__main__':  #test code
    img_file = '/disk_data/data/ACDC_2019/Testing/h46_w21.jpg'
    save_flename = '/disk_data/data/ACDC_2019/Testing/h46_w21_pred.jpg'

    # tensor1 = img_to_tensor(img_file, image_shape=(512, 512))
    tensor1 = img_to_tensor(img_file)
    img1 = tensor_to_image(tensor1, '/disk_data/data/ACDC_2019/Testing/h46_w21_restore.jpg')

    print('OK')