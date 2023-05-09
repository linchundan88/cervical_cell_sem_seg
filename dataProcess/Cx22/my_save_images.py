import scipy.io
import cv2
from pathlib import Path

file_mat = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/generator/ImageDataSet.mat'
dir_save = '/disk_data/data/carvical/Cx22/training'

file_mat = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Test/generator/ImageDataSet.mat'
dir_save = '/disk_data/data/carvical/Cx22/testing'


mat = scipy.io.loadmat(file_mat)

for i in range(mat['ImageDataSet'].shape[0]):
    img1 = mat['ImageDataSet'][i, 0]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{dir_save}/a{i}.jpg', img1)


print('OK')