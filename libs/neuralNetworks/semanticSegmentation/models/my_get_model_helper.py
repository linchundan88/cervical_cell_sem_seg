import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus
from libs.neuralNetworks.semanticSegmentation.models.unet.u_net_variants import R2U_Net, AttU_Net, R2AttU_Net
from libs.neuralNetworks.semanticSegmentation.models.unet.u2_net import U2NET, U2NETP
from libs.neuralNetworks.semanticSegmentation.models.unet_plus3.UNet_3Plus import UNet_3Plus
from libs.neuralNetworks.semanticSegmentation.models.unet_plus3.UNet_2Plus import UNet_2Plus
# from libs.neuralNetworks.semanticSegmentation.models.unet_plus3.UNet import UNet
from libs.neuralNetworks.semanticSegmentation.models.u_net.unet_model import UNet
from libs.neuralNetworks.semanticSegmentation.models.transunet.transunet import TransUNet


def get_model_smp(model_type, in_channels=3, n_classes=1, encoder_name='resnet34', encoder_weights='imagenet', model_file=None):

    dict_param = dict(encoder_weights=encoder_weights, encoder_depth=5, encoder_name=encoder_name,
                      classes=n_classes, in_channels=in_channels, activation=None)
    if model_type == 'Unet':
        model = Unet(**dict_param)
    if model_type == 'UnetPlusPlus':
        model = UnetPlusPlus(**dict_param)
    if model_type == 'DeepLabV3':
        model = DeepLabV3(**dict_param)
    if model_type == 'DeepLabV3Plus':
        model = DeepLabV3Plus(**dict_param)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def get_model_transunet(img_dim = 128, in_channels = 3, out_channels = 128,
    head_num = 4, mlp_dim = 512, block_num = 8, patch_dim = 16, n_classes=1,  model_file=None):

    model = TransUNet(img_dim=img_dim, in_channels=in_channels, out_channels=out_channels, head_num=head_num,
                          mlp_dim=mlp_dim, block_num=block_num, patch_dim=patch_dim, class_num=n_classes)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model



def get_model_trans_unet(vit_name=None, img_size=488, n_classes=1, n_skip=3,  model_file=None):

    from libs.neuralNetworks.semanticSegmentation.models.trans_unet.vit_seg_modeling import VisionTransformer as ViT_seg
    from libs.neuralNetworks.semanticSegmentation.models.trans_unet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = n_skip

    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def get_model_others(model_type, in_channels=3, n_classes=1, model_file=None):

    if model_type == 'UNet_o':
        model = UNet(n_channels=in_channels, n_classes=n_classes)

    if model_type == 'UNet_2Plus':
        model = UNet_2Plus(in_channels=in_channels, n_classes=n_classes)
    if model_type == 'Unet_3P':
        model = UNet_3Plus(in_channels=in_channels, n_classes=n_classes)

    if model_type == 'R2U_Net':
        model = R2U_Net(img_ch=in_channels, output_ch=n_classes)
    if model_type == 'AttU_Net':
        model = AttU_Net(img_ch=in_channels, output_ch=n_classes)
    if model_type == 'R2AttU_Net':
        model = R2AttU_Net(img_ch=in_channels, output_ch=n_classes)


    if model_type == 'U2_NET':
        model = U2NET(in_ch=in_channels, out_ch=n_classes)
    if model_type == 'U2NETP':
        model = U2NETP(in_ch=in_channels, out_ch=n_classes)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


    # model = Unet('vgg16', encoder_weights='imagenet', encoder_depth=4, decoder_channels=[512, 256, 128, 64],  activation=None)
    # model = smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, encoder_depth=4,
    #                  decoder_channels=[128, 64, 32, 16], activation=None)


