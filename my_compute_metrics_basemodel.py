
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='cyto_ins')   # cyto_ins nuc_ins
parser.add_argument('--model_type', default='Transunet')
parser.add_argument('--encoder_weights', default='none')   # imagenet none
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model_smp
from libs.neuralNetworks.semanticSegmentation.dataset.my_dataset import Dataset_SEM_SEG, get_dataset
from torch.utils.data import DataLoader
from libs.neuralNetworks.semanticSegmentation.my_predict_helper import predict_multi_models
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_sen, get_spe, get_iou, get_dice
import torch

# task_type:cyto_ins, model_type: Unet_resnet34, encoder_weights:none
csv_test = Path(__file__).resolve().parent / 'datafiles' / f'{args.task_type}' / 'test.csv'
patch_h, patch_w = (512, 512)

if args.encoder_weights == 'imagenet':
    path_models = Path(__file__).resolve().parent / 'trained_models' / 'imagenet_encoder' / args.task_type
else:
    path_models = Path(__file__).resolve().parent / 'trained_models' / 'without_pretrained' / args.task_type


model_dicts = []

if args.encoder_weights == 'imagenet':
    if args.task_type == 'nuc_ins':
        if args.model_type == 'Unet_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.044_epoch12.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'Unet_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.042_epoch15.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.026_epoch16.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.033_epoch16.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)

        elif args.model_type == 'DeepLabV3_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.045_epoch15.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.041_epoch15.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet50', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.053_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.037_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet50', model_file=model_file1)

    if args.task_type == 'cyto_ins':
        if args.model_type == 'Unet_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.058_epoch16.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'Unet_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.051_epoch15.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.055_epoch16.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.053_epoch17.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)

        elif args.model_type == 'DeepLabV3_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.079_epoch17.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.077_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet50', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.066_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.066_epoch17.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet50', model_file=model_file1)

elif args.encoder_weights.strip() == '' or args.encoder_weights.strip() == 'none':
    if args.task_type == 'nuc_ins':
        if args.model_type == 'Unet_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.079_epoch16.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'Unet_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.043_epoch15.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.034_epoch15.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.035_epoch14.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)

        elif args.model_type == 'DeepLabV3_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.059_epoch17.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.057_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet50', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.074_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.065_epoch17.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet50', model_file=model_file1)

    if args.task_type == 'cyto_ins':
        if args.model_type == 'Unet_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.126_epoch16.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'Unet_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.106_epoch16.pth'
            model1 = get_model_smp(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.094_epoch17.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'UnetPlusPlus_densenet121':
            model_file1 = path_models / args.model_type / 'valid_loss_0.087_epoch15.pth'
            model1 = get_model_smp(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)

        elif args.model_type == 'DeepLabV3_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.106_epoch17.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.113_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3', encoder_name='resnet50', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet34':
            model_file1 = path_models / args.model_type / 'valid_loss_0.095_epoch16.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet34', model_file=model_file1)
        elif args.model_type == 'DeepLabV3Plus_resnet50':
            model_file1 = path_models / args.model_type / 'valid_loss_0.094_epoch14.pth'
            model1 = get_model_smp(model_type='DeepLabV3Plus', encoder_name='resnet50', model_file=model_file1)


        elif args.model_type == 'Transunet':
            from self_attention_cv.transunet import TransUnet
            model_file1 = path_models / args.model_type / 'valid_loss_0.545_epoch2.pth'
            model1 = TransUnet(img_dim=patch_h, in_channels=3, vit_blocks=8, vit_dim_linear_mhsa_block=512, classes=1)
        elif args.model_type == 'Segformer':
            from libs.neuralNetworks.semanticSegmentation.models.segformer.segformer import Segformer
            model_file1 = path_models / args.model_type / 'valid_loss_0.157_epoch16.pth'
            model1 = Segformer(
                dims=(32, 64, 160, 256),  # dimensions of each stage
                heads=(1, 2, 5, 8),  # heads of each stage
                ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
                reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
                num_layers=2,  # num layers of each stage
                decoder_dim=256,  # decoder dimension
                num_classes=1
            )
            state_dict = torch.load(model_file1, map_location='cpu')
            model1.load_state_dict(state_dict)

model_dict1 = dict(model=model1, use_amp=True, model_weight=1, image_shape=(patch_h, patch_w), batch_size=64)
model_dicts.append(model_dict1)


data_loaders = []
model_dict = model_dicts[0]
ds_test = Dataset_SEM_SEG(csv_file=csv_test, image_shape=model_dict['image_shape'], mask_thresholds= ((100, 255),))
dataloader_test = DataLoader(ds_test, batch_size=model_dict['batch_size'], num_workers=8, pin_memory=True)
data_loaders.append(dataloader_test)

ds_test = Dataset_SEM_SEG(csv_file=csv_test, image_shape=model_dict['image_shape'], mask_thresholds= ((100, 255),))
images, labels = get_dataset(ds_test)

list_outputs, ensemble_outputs = predict_multi_models(model_dicts, data_loaders, mode='DP')


threshold=0.5
TP_num, TN_num, FP_num, FN_num = get_confusion_matrix(ensemble_outputs, labels, threshold=threshold)
sen = get_sen(ensemble_outputs, labels, threshold=threshold)
spe = get_spe(ensemble_outputs, labels, threshold=threshold)
dice = get_dice(ensemble_outputs, labels, threshold=threshold)


print(f'DICE:{round(dice, 4)}, SEN:{round(sen, 4)}, SPE:{round(spe, 4)}')





