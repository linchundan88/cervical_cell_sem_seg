
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='cyto_ins')   # nuc_ins cyto_ins

args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model_smp
from libs.neuralNetworks.semanticSegmentation.my_predict_helper import predict_one_file
import cv2


img_file_source = 'test_images/a90.jpg'
img_file_result = f'a90_result_{args.task_type}.jpg'

patch_h, patch_w = (512, 512)

model_dicts = []
path_models = Path(__file__).resolve().parent / 'trained_models' / 'imagenet_encoder'

print('loading models...')

if args.task_type == 'nuc_ins':
    model_file1 = path_models / args.task_type / 'Unet_resnet34' / 'valid_loss_0.044_epoch12.pth'
    model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'Unet_densenet121' / 'valid_loss_0.042_epoch15.pth'
    model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'UnetPlusPlus_resnet34' / 'valid_loss_0.026_epoch16.pth'
    model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'UnetPlusPlus_densenet121' / 'valid_loss_0.033_epoch16.pth'
    model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

if args.task_type == 'cyto_ins':
    model_file1 = path_models / args.task_type / 'Unet_resnet34' / 'valid_loss_0.058_epoch16.pth'
    model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'Unet_densenet121' / 'valid_loss_0.051_epoch15.pth'
    model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'UnetPlusPlus_resnet34' / 'valid_loss_0.055_epoch16.pth'
    model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / args.task_type / 'UnetPlusPlus_densenet121' / 'valid_loss_0.053_epoch17.pth'
    model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
    model_dicts.append(model_dict1)

print('loading models completed.')


list_outputs, ensemble_outputs = predict_one_file(img_file_source, model_dicts, save_file=img_file_result)


# outputs = cv2.imread('a0_result.jpg', cv2.IMREAD_GRAYSCALE)


# from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_sen, get_spe, get_iou, get_dice
# labels = cv2.imread('a0_nuc_ins.jpg', cv2.IMREAD_GRAYSCALE)
# threshold=127
# TP_num, TN_num, FP_num, FN_num = get_confusion_matrix(outputs, labels, threshold=threshold)
# sen = get_sen(outputs, labels, threshold=threshold)
# spe = get_spe(outputs, labels, threshold=threshold)
# get_iou = get_iou(outputs, labels, threshold=threshold)
# get_dice = get_dice(outputs, labels, threshold=threshold)




