
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='nuc_ins')   # cyto_ins nuc_ins
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model_smp
from libs.neuralNetworks.semanticSegmentation.dataset.my_dataset import Dataset_SEM_SEG, get_dataset
from torch.utils.data import DataLoader
from libs.neuralNetworks.semanticSegmentation.my_predict_helper import predict_multi_models
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_acc, get_sen, get_spe, get_iou, get_dice
from libs.my_helper_ststistics import get_cf_bootstraping


csv_test = Path(__file__).resolve().parent / 'datafiles' / f'{args.task_type}' / 'test.csv'
patch_h, patch_w = (512, 512)


model_dicts = []
path_models = Path(__file__).resolve().parent / 'trained_models' / 'imagenet_encoder'

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



data_loaders = []
for model_dict in model_dicts:
    ds_test = Dataset_SEM_SEG(csv_file=csv_test, image_shape=model_dict['image_shape'], mask_thresholds= ((100, 255),))
    dataloader_test = DataLoader(ds_test, batch_size=model_dict['batch_size'], num_workers=8, pin_memory=True)
    data_loaders.append(dataloader_test)

ds_test = Dataset_SEM_SEG(csv_file=csv_test, image_shape=model_dict['image_shape'], mask_thresholds= ((100, 255),))
images, labels = get_dataset(ds_test)

list_outputs, ensemble_outputs = predict_multi_models(model_dicts, data_loaders, mode='DP')

ensemble_outputs = ensemble_outputs.flatten()
labels = labels.flatten()

threshold=0.5
TP_num, TN_num, FP_num, FN_num = get_confusion_matrix(ensemble_outputs, labels, threshold=threshold)
# acc = get_acc(ensemble_outputs, labels, threshold=threshold)
sen = get_sen(ensemble_outputs, labels, threshold=threshold)
spe = get_spe(ensemble_outputs, labels, threshold=threshold)
# iou = get_iou(ensemble_outputs, labels, threshold=threshold)
dice = get_dice(ensemble_outputs, labels, threshold=threshold)
print(f'SEN:{round(sen, 4)}, SPE:{round(spe, 4)}, DICE:{round(dice, 4)}')


sampling_times = 500
for cf_level in [0.95]:
    print(f'confidence level:{cf_level}')
    # acc_cf = get_cf_bootstraping(get_acc, ensemble_outputs, labels, sampling_times=500, cf_level=cf_level)
    # print(acc_cf)
    sen_cf = get_cf_bootstraping(get_sen, ensemble_outputs, labels, sampling_times=sampling_times, cf_level=cf_level)
    print('confidence of sen:', sen_cf)
    spe_cf = get_cf_bootstraping(get_spe, ensemble_outputs, labels, sampling_times=sampling_times, cf_level=cf_level)
    print('confidence of spe:', spe_cf)
    dice_cf = get_cf_bootstraping(get_dice, ensemble_outputs, labels, sampling_times=sampling_times, cf_level=cf_level)
    print('confidence of dice:', dice_cf)


print('OK')
