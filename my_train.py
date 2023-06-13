'''The training file, this code can be invoked by my_train.sh through the command line.'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_devices', default='0,1')
parser.add_argument('--task_type', default='nuc_ins')  # 'cyto_ins'pos_weight:1, 'nuc_ins' pos_weight:8   as for semantic segmentation nuc_clumps' is the same as 'nuc_ins'
parser.add_argument('--model_type', default='Segformer')  #Unet UnetPlusPlus DeepLabV3 DeepLabV3Plus Transunet Segformer  Trans_unetUNet_o Unet_3P UNet_2Plus R2U_Net
parser.add_argument('--encoder_name', default='')  # resnet34 resnet50 densenet121 vgg13 vgg13_bn  resnet34 mobilenet_v2 timm-mobilenetv3_small_100 timm-mobilenetv3_large_075
parser.add_argument('--encoder_weights', default='imagenet')  # imagenet, '', 'none'
parser.add_argument('--vit_name', default='ViT-B_16')  # used by Trans_unet. ViT-B_16 ViT-B_32 ViT-L_16 ViT-L_32 ViT-H_14  please see CONFIGS in vit_seg_modeling.py
parser.add_argument('--image_shape', nargs='+', type=int, default=(512, 512))  #patch_size (512, 512),  random cropping (448,448)
parser.add_argument('--loss_function', default='bce')   #dice, bce, softbce, combine_dice_bce
parser.add_argument('--pos_weight', type=float, default=8)  # only used for bce loss, 1 for cyto_ins and 8 for nuc_ins
parser.add_argument('--smooth_factor', type=float, default='0.1')  # only used for softbce loss
parser.set_defaults(use_amp=True)  # automatic mixed precision
parser.add_argument('--no-amp', dest='use_amp', action='store_false')
parser.set_defaults(compile=False)
parser.add_argument('--compile', dest='compile', action='store_true')
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)  # batch size of DDP should equal batch size of DDP / gpu number:
parser.add_argument('--weight_decay', type=float, default=0)  #L2 regularization, 2e-5 is a good choice for U-Net Resnet34
parser.add_argument('--epochs_num', type=int, default=18)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_model_dir', default='/disk_code/code/cervical_cell_sem_seg/trained_models/2023_5_8/')
parser.add_argument('--parallel_mode', choices=['DP', 'DDP'], default='DP')   # DP:Data Parallel,  DDP:Distributed Data Data Parallel
parser.add_argument('--sync_bn', default=False)  # distributed data parallel whether using convert_sync_batchnorm
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices   # setting  GPUs, must before import torch
import torch
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model_smp, get_model_trans_unet, get_model_transunet, get_model_others
from libs.neuralNetworks.semanticSegmentation.dataset.my_dataset import Dataset_SEM_SEG
import torch.optim as optim
# import torch_optimizer as optim_plus
from torch.optim.lr_scheduler import MultiStepLR
import albumentations as A
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.soft_bce import SoftBCEWithLogitsLoss
from libs.neuralNetworks.semanticSegmentation.losses.my_loss_sem_seg import Tversky_loss
from libs.neuralNetworks.semanticSegmentation.losses.my_ce_dice import CE_Dice_combine_Loss
# from libs.neuralNetworks.semanticSegmentation.losses.my_loss_sem_seg import DiceLoss
from libs.neuralNetworks.semanticSegmentation.my_train_helper import train_DP, train_DDP
import torch.multiprocessing as mp
from munch import Munch
from libs.neuralNetworks.semanticSegmentation.my_train_helper import draw_loss_graph
import pickle


if __name__ == '__main__':

    #region prepare dataset including image augmentation
    path_csv = Path(__file__).resolve().parent / 'datafiles'
    csv_train = path_csv / f'{args.task_type}' / 'train.csv'
    csv_valid = path_csv / f'{args.task_type}' / 'valid.csv'

    mask_thresholds = ((100, 255),)   # support multi-class, or single value threshold ((127),)
    # IMAGE_CYTO_INS = 128    IMAGE_CYTO_CLUMPS = 128 + 64    IMAGE_NUC_INS = 128 + 96     IMAGE_NUC_CLUMPS = 255
    num_classes = len(mask_thresholds)

    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomCrop(width=args.image_shape[1], height=args.image_shape[0]),  #512->448?
        A.RandomRotate90(p=0.8),
        A.ShiftScaleRotate(p=0.8, rotate_limit=(-10, 10)),  #shift_limit=0.05, scale_limit=0.1
        # A.Affine(scale=0.1, rotate=10, translate_percent=0.1),
        A.RandomBrightnessContrast(p=0.8, brightness_limit=0.2, contrast_limit=0.2),
        # A.ColorJitter(p=0.8, saturation=0.2, hue=0.05),  # brightness=0.2, contrast=0.2,
        A.HueSaturationValue(p=0.8, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10),
        A.GaussianBlur(p=0.6, blur_limit=(3, 5)),
        A.Resize(args.image_shape[0], args.image_shape[1]),  #(height,weight)  if random cropping patch size is different from ...
    ])

    ds_train = Dataset_SEM_SEG(csv_file=csv_train, transform=transform_train, mask_thresholds=mask_thresholds)  #image_shape=args.image_shape,
    ds_valid = Dataset_SEM_SEG(csv_file=csv_valid, image_shape=args.image_shape, mask_thresholds=mask_thresholds)

    #endregion

    # region defining models
    if args.model_type in ['Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus']:  # encoder_weights: 'imagenet'  or None without pretrained
        if args.encoder_weights == 'imagenet':
            encode_weights = 'imagenet'
        if args.encoder_weights.strip() == '' or args.encoder_weights.strip() == 'none':
            encode_weights = None
        model = get_model_smp(model_type=args.model_type, encoder_name=args.encoder_name, encoder_weights=encode_weights, in_channels=3, n_classes=num_classes)
    elif args.model_type == 'Transunet':
        from self_attention_cv.transunet import TransUnet
        model = TransUnet( img_dim=args.image_shape[0], in_channels=3, vit_blocks=8, vit_dim_linear_mhsa_block=512, classes=num_classes)
    # elif args.model_type == 'Trans_unet':
    #     model = get_model_trans_unet(vit_name = args.vit_name, img_size = args.image_shape[0], n_classes=num_classes, n_skip = 3)
    elif args.model_type == 'Segformer':
        from libs.neuralNetworks.semanticSegmentation.models.segformer.segformer import Segformer
        model = Segformer(
            dims=(32, 64, 160, 256),  # dimensions of each stage
            heads=(1, 2, 5, 8),  # heads of each stage
            ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
            reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
            num_layers=2,  # num layers of each stage
            decoder_dim=256,  # decoder dimension
            num_classes=num_classes
        )
    else:
        model = get_model_others(model_type=args.model_type, in_channels=3, n_classes=num_classes)

    if args.compile:
        model = torch.compile(model)

    #endregion

    #region loss function, optimizer and  scheduler.
    if args.loss_function == 'dice':
        criterion = DiceLoss(mode='binary', from_logits=True)  # segmentation_models_pytorch loss function
    if args.loss_function == 'bce':
        pos_weight = torch.FloatTensor(torch.tensor([float(args.pos_weight)]))
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.loss_function == 'combine_dice_bce':
        pos_weight = torch.FloatTensor(torch.tensor([float(args.pos_weight)]))
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        criterion = CE_Dice_combine_Loss(weight_bce=1, weight_dice=1, pos_weight=pos_weight)
    if args.loss_function == 'softbce':
        pos_weight = torch.FloatTensor(torch.tensor([float(args.pos_weight)]))
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        criterion = SoftBCEWithLogitsLoss(pos_weight=pos_weight, smooth_factor=args.smooth_factor)
    if args.loss_function == 'tversky':
        # versky index can also be seen as an generalization of Dices coefficient.
        # It adds a weight to FP (false positives) and FN (false negatives) with the help of β coefficient
        criterion = Tversky_loss(alpha=0.5, beta=0.5)
        #from monai.losses import FocalLoss
    # criterion = DiceLoss(activation='sigmoid')  #My loss function

    if args.parallel_mode == 'DDP':
        n_gpus = torch.cuda.device_count()
        args.lr *= n_gpus
        args.batch_size = int(args.batch_size / n_gpus)  # total batch size,  batch size per gpu

    epochs_num = args.epochs_num
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs_num // 4, eta_min=0)  #T_max: half of one circle
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)
    scheduler_mode = 'epoch'
    scheduler = MultiStepLR(optimizer, milestones = [int(epochs_num * 0.4), int(epochs_num * 0.8)], gamma=0.1)

    #Munch is better than Dict. The contents of it can be accessed by dot operator or string name.
    path_save = Path(args.save_model_dir) / args.task_type / f'{args.model_type}_{args.encoder_name}'
    train_config = Munch({
        'model': model, 'ds_train': ds_train, 'criterion': criterion, 'ds_valid': ds_valid,
        'epochs_num': args.epochs_num, 'optimizer': optimizer,  'scheduler': scheduler, 'scheduler_mode': scheduler_mode,
        'batch_size': args.batch_size, 'num_workers': args.num_workers, 'use_amp': args.use_amp, 'sync_bn': False,
        'save_model_dir': path_save, 'losses_pkl': path_save / 'losses.pkl'
    })

    #endregion

    if args.parallel_mode == 'DP':
        train_DP(train_config)
    else:
        n_gpus = torch.cuda.device_count()
        mp.spawn(train_DDP, args=(n_gpus, train_config), nprocs=n_gpus, join=True)

    list_losses = pickle.load(open(train_config.losses_pkl, 'rb'))  # the multi processer function can not return values
    draw_loss_graph(list_losses, path_save / 'losses.png')
    print('OK')





