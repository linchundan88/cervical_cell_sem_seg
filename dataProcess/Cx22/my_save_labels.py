'''
Beginning at release 7.3 of Matlab, mat files are actually saved using the HDF5 format by default
'''


import h5py
import numpy as np
import cv2
import pickle
from pathlib import Path


def get_bbox(file_mat):
    with h5py.File(file_mat, 'r') as f:
        # print(list(f.keys()))
        dset = f['cyto_ins_bbox']

        image_num = dset.size
        list_images = []

        for img_index in range(image_num):
            list_bbox = []

            _, object_nums = f[dset[0][img_index]].shape
            for object_index in range(object_nums):
                bbox = []
                bbox.append(f[dset[0][img_index]][0, object_index])
                bbox.append(f[dset[0][img_index]][1, object_index])
                bbox.append(f[dset[0][img_index]][2, object_index])
                bbox.append(f[dset[0][img_index]][3, object_index])
            
                list_bbox.append(bbox)

            list_images.append(list_bbox)

        return list_images


def get_ins(file_mat, dataset_name):
    with h5py.File(file_mat, 'r') as f:
        dset = f[dataset_name]
        image_num = dset.size
        list_image_masks_all = []
        list_contours_all = []

        for img_index in range(image_num):
            list_image_masks = []
            list_contours = []

            _, object_nums = f[dset[0, img_index]].shape
            for object_index in range(object_nums):
                img_mask = f[f[dset[0, img_index]][0, object_index]]
                img_mask = np.array(img_mask)
                list_image_masks.append(img_mask)

                img_mask_gray = img_mask * 255
                contours, _ = cv2.findContours(img_mask_gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                list_contours.append(contours)

            list_image_masks_all.append(list_image_masks)
            list_contours_all.append(list_contours)

    return list_image_masks_all, list_contours_all



# dir_save = '/disk_data/data/carvical/Cx22/patches/training/masks'
dir_save = '/disk_data/data/carvical/Cx22/patches/test/masks'
(Path(dir_save) / 'imgs').mkdir(exist_ok=True, parents=True)

# file_mat = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/cyto/cyto_ins_bbox.mat'
# list_bboxex = get_bbox(file_mat)

# file_cyto_clumps = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/cyto/cyto_clumps.mat'
# file_cyto_ins = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/cyto/cyto_ins.mat'
# file_nuc_clumps = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/nuc/nuc_clumps.mat'
# file_nuc_ins = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Train/nuc/nuc_ins.mat'

file_cyto_clumps = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Test/cyto/cyto_clumps.mat'
file_cyto_ins = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Test/cyto/cyto_ins.mat'
file_nuc_clumps = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Test/nuc/nuc_clumps.mat'
file_nuc_ins = '/disk_data/data/carvical/Cx22/Cx22-main/Cx22-Multi-Test/nuc/nuc_ins.mat'

IMAGE_CYTO_INS = 128
IMAGE_CYTO_CLUMPS = 128+64
IMAGE_NUC_INS = 128 + 96
IMAGE_NUC_CLUMPS = 255


for seg_type in ['cyto_clumps', 'cyto_ins', 'nuc_clumps', 'nuc_ins']:
    if seg_type == 'cyto_clumps':
        file_mat = file_cyto_clumps
        pixel_value = IMAGE_CYTO_CLUMPS
    if seg_type == 'cyto_ins':
        file_mat = file_cyto_ins
        pixel_value = IMAGE_CYTO_INS
    if seg_type == 'nuc_clumps':
        file_mat = file_nuc_clumps
        pixel_value = IMAGE_NUC_CLUMPS
    if seg_type == 'nuc_ins':
        file_mat = file_nuc_ins
        pixel_value = IMAGE_NUC_INS

    list_mask, list_contour = get_ins(file_mat, dataset_name=seg_type)
    for i, image_masks in enumerate(list_mask):
        list_masks = []
        for j, image_mask in enumerate(image_masks):
            image_mask = image_mask.transpose()
            list_masks.append(image_mask)

            filename_mask = f'{dir_save}/imgs/a{i}_{seg_type}_{j}.jpg'
            print(filename_mask)
            cv2.imwrite(filename_mask, image_mask * pixel_value)

        img_mask_all = np.zeros((image_mask.shape[0], image_mask.shape[1]), dtype=int)
        for mask in list_masks:
            img_mask_all[mask == 1] = pixel_value

        filename_mask = f'{dir_save}/a{i}_{seg_type}.jpg'
        print(filename_mask)
        cv2.imwrite(filename_mask, img_mask_all)


list_mask_cyto_clumps, list_contour_cyto_clumps = get_ins(file_cyto_clumps, dataset_name='cyto_clumps')
list_mask_cyto_ins, list_contour_cyto_ins = get_ins(file_cyto_ins, dataset_name='cyto_ins')
list_mask_nuc_clumps, list_contour_nuc_clumps = get_ins(file_nuc_clumps, dataset_name='nuc_clumps')
list_mask_nuc_ins, list_contour_nuc_ins = get_ins(file_nuc_ins, dataset_name='nuc_ins')

for i, (mask_cyto_clumps, mask_cyto_ins, mask_nuc_clumps, mask_nuc_ins) in \
        enumerate(zip(list_mask_cyto_clumps, list_mask_cyto_ins, list_mask_nuc_clumps, list_mask_nuc_ins)):
    dict1 = {'cyto_clumps': mask_cyto_clumps, 'cyto_ins': mask_cyto_ins, 'nuc_clumps': mask_nuc_clumps, 'nuc_ins': mask_nuc_ins}
    with open(f'{dir_save}/a{i}_masks.pkl', 'wb') as f:
        pickle.dump(dict1, f)

for i, (contour_cyto_clumps, contour_cyto_ins, contour_nuc_clumps, contour_nuc_ins) in \
        enumerate(zip(list_contour_cyto_clumps, list_contour_cyto_ins, list_contour_nuc_clumps, list_contour_nuc_ins)):
    dict1 = {'cyto_clumps': contour_cyto_clumps, 'cyto_ins': contour_cyto_ins, 'nuc_clumps': contour_nuc_clumps, 'nuc_ins': contour_nuc_ins}
    with open(f'{dir_save}/a{i}_contours.pkl', 'wb') as f:
        pickle.dump(dict1, f)


'''
for i, (image_masks_cyto, image_masks_nuc) in enumerate(zip(list_image_cyto, list_image_nuc)):
    for j, (image_mask_cyto, image_mask_nuc) in enumerate(zip(image_masks_cyto, image_masks_nuc)):
        image_mask_cyto = image_mask_cyto.transpose()
        image_mask_cyto = image_mask_cyto * IMAGE_CYTO
        filename_mask_cyto = f'{dir_save}/imgs/a{i}_cyto_{j}.jpg'
        cv2.imwrite(filename_mask_cyto, image_mask_cyto)

        image_mask_nuc = image_mask_nuc.transpose()
        image_mask_nuc = image_mask_nuc * IMAGE_NUC
        filename_mask_nuc = f'{dir_save}/imgs/a{i}_nuc_{j}.jpg'
        cv2.imwrite(filename_mask_nuc, image_mask_nuc)

        # the indexes of cyto and nuc are not in one-to-one correspondence
        # image_mask_cyto[image_mask_nuc == 1] = 0
        # image_mask = image_mask_cyto + image_mask_nuc
        # filename_mask = f'{dir_save}/imgs/a{i}_{j}.jpg'
'''


print('OK')




