'''
predict_single_model: data_loader, combine results of multiple batches.
predict_multi_models:  call predict_single_model multiple times and do weighted averaging.
predict_one_patch: predict a single patch.
'''

import numpy as np
import cupy as cp
import cv2
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from libs.neuralNetworks.semanticSegmentation.dataset.my_dataset import Dataset_SEM_SEG
from libs.images.my_img_to_tensor import img_to_tensor
from tqdm import tqdm
import time
from torch.cuda.amp import autocast
import gc
from libs.images.my_img import remove_small_object


@torch.inference_mode()
def predict_single_model(model, data_loader, use_amp, mode='DP', sync_bn=False, scale_ratio=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if mode =='DP':
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    if mode == 'DDP' and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.eval()

    list_outputs = []
    for batch_idx, inputs in enumerate(tqdm(data_loader)):
        if isinstance(inputs, list):  # both images and masks
            inputs, masks = inputs
        inputs = inputs.to(device)
        with autocast(enabled=use_amp):
            outputs = model(inputs)  #B,C,H,W
            outputs = torch.sigmoid(outputs)

        outputs = outputs.cpu().numpy().astype(np.float16)  # B, C, H, W
        assert scale_ratio in [1/8, 1/4, 1/2, 1], f'scale ratio:{scale_ratio} error!'
        if scale_ratio != 1:   #for big WSI, predicted results can not fit into memory.
            slice_index = int(1 // scale_ratio)
            outputs = outputs[:, :, ::slice_index, ::slice_index]
        list_outputs.append(outputs)

    outputs = np.vstack(list_outputs)  # Equivalent to np.concatenate(list_outputs, axis=0)

    return outputs


#multi models using the different data loaders image sizes
def predict_multi_models(model_dicts, data_loaders, use_cupy=False, mode='DP', sync_bn=False, scale_ratio=1):
    list_outputs = []

    for data_loader, model_dict in zip(data_loaders, model_dicts):
        outputs = predict_single_model(model_dict['model'], data_loader, model_dict['use_amp'], mode, sync_bn, scale_ratio)
        list_outputs.append(outputs)

        # del mdoel
        gc.collect()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

    for index, (model_dict, outputs) in enumerate(zip(model_dicts, list_outputs)):
        if use_cupy:
            outputs = cp.asarray(outputs, cp.float16)

        if model_dict['model_weight'] != 1:
            outputs = outputs * model_dict['model_weight']

        if index == 0:  # if 'probs_total' not in locals().keys():
            ensemble_outputs = outputs
            total_weights = model_dict['model_weight']
        else:
            ensemble_outputs += outputs
            total_weights += model_dict['model_weight']

    if total_weights != 1:
        ensemble_outputs /= total_weights

    if use_cupy:
        ensemble_outputs = cp.asnumpy(ensemble_outputs)
        gc.collect()
        torch.cuda.empty_cache()

    return list_outputs, ensemble_outputs


def predict_one_file(img_file, model_dicts, save_file=None, model_convert_gpu=True, small_object_threshold=10):
    assert Path(img_file).exists(), 'fine not found!'

    list_outputs = []

    for index, model_dict in enumerate(model_dicts):
        model = model_dict['model']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_convert_gpu and torch.cuda.device_count() > 0:
            model.to(device)  # model.cuda()
        model.eval()

        if index == 0:  # Reduce the number of reading image files
            image_shape = model_dict['image_shape']
            inputs = img_to_tensor(img_file, image_shape=model_dict['image_shape'])
            inputs = inputs.to(device)
        elif image_shape != model_dict['image_shape']:
            inputs = img_to_tensor(img_file, image_shape=model_dict['image_shape'])
            inputs = inputs.to(device)

        with torch.inference_mode():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

        outputs = outputs.cpu().numpy()
        list_outputs.append(outputs)

    for index, (model_dict, outputs) in enumerate(zip(model_dicts, list_outputs)):
        if index == 0:  # if 'probs_total' not in locals().keys():
            ensemble_outputs = outputs * model_dict['model_weight']
            total_weights = model_dict['model_weight']
        else:
            ensemble_outputs += outputs * model_dict['model_weight']
            total_weights += model_dict['model_weight']

    ensemble_outputs /= total_weights

    if save_file:
        pred_mask = ensemble_outputs[0, 0, :, :]  # (N,C,H,W)
        _, img_thres = cv2.threshold(pred_mask, 0.5, 255, cv2.THRESH_BINARY)
        if remove_small_object is not None:
            img_thres = remove_small_object(img_thres, small_object_threshold)

        cv2.imwrite(save_file, img_thres)

    return list_outputs, ensemble_outputs



# if __name__ == '__main__':
#     print('OK')