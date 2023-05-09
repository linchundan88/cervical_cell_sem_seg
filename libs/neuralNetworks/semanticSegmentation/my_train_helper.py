'''
    my_train_DP: data parallel
    train_one_epoch: used by both DP and DDP training.
    validate: used by both DP and DDP training.
    draw_loss_graph:
'''

import warnings
import numpy as np
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics #CPU, slow
from tqdm import tqdm
import logging
from datetime import datetime
import gc
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle


def train_DP(config):
    model1 = config.model
    ds_train = config.ds_train
    ds_valid = config.ds_valid
    epochs_num = config.epochs_num
    optimizer = config.optimizer
    criterion = config.criterion
    scheduler = config.scheduler
    scheduler_mode = config.scheduler_mode    
    use_amp = config.use_amp
    batch_size = config.batch_size
    num_workers = config.num_workers
    save_model_dir = config.save_model_dir
    save_model_dir.mkdir(parents=True, exist_ok=True)
    losses_pkl = config.losses_pkl

    list_loss_history = []  # using list_loss_history to draw the line of training and validation loss
    logging.basicConfig(filename=save_model_dir / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model1.to(device)
    if torch.cuda.device_count() > 1:
        model1 = nn.DataParallel(config.model)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    for epoch in range(epochs_num):
        loss_train, tp_train, tn_train, fp_train, fn_train = _train_one_epoch(model1, loader_train, criterion, epoch, optimizer,
                                                                              scheduler, scheduler_mode, use_amp)

        acc_train, sen_train, spe_train, dice_train = get_metrics(tp_train, tn_train, fp_train, fn_train)
        print(f'training epoch{epoch} metrics:')
        print(f'losses:{loss_train:.3f}')
        print(f'acc:{acc_train:.3f}, sen:{sen_train:.3f}, spe:{spe_train:.3f}')
        print(f'dice:{dice_train:.3f}')

        print(f'epoch:{epoch} compute validation dataset...')
        dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        loss_valid, tp_valid, tn_valid, fp_valid, fn_valid = _validate_one_epoch(model1, dataloader_valid, criterion, use_amp=use_amp)

        acc_valid, sen_valid, spe_valid, dice_valid = get_metrics(tp_valid, tn_valid, fp_valid, fn_valid)
        print(f'validation metrics:')
        print(f'losses:{loss_valid:.3f}')
        print(f'acc:{acc_valid:.3f}, sen:{sen_valid:.3f}, spe:{spe_valid:.3f}')
        print(f'dice:{dice_valid:.3f}')

        list_loss_history.append([loss_train, loss_valid])
        save_model_file = save_model_dir / f'valid_loss_{round(loss_valid, 3)}_epoch{epoch}.pth'
        try:
            state_dict = model1.module.state_dict()
        except AttributeError:
            state_dict = model1.state_dict()
        print('save model:', save_model_file)
        torch.save(state_dict, save_model_file)

    pickle.dump(list_loss_history, open(losses_pkl, 'wb'))
    clear_gpu_cache()


def _train_one_epoch(model, dataloader, criterion, epoch, optimizer, scheduler, scheduler_mode, use_amp,  rank=None):
    if rank is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda', rank)  # distributed data parallel
    model.train()
    scaler = GradScaler(enabled=use_amp)
    epoch_loss, epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0, 0

    if rank is None or rank==0:
        dataloader = tqdm(dataloader, desc=f'Training epoch {epoch}')  # the processor other than rank0 do not show progressbar.
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)  # the datatype of masks is int.

        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logging.info(f'epoch:{epoch} training batch:{batch_idx}, losses:{loss.item():3}')

        epoch_loss += loss.item()  # loss function setting reduction='mean'

        #show performance indicator during training
        with torch.inference_mode():
            outputs = F.sigmoid(outputs)

        # the WSI can be so huge that its results can not be saved into memory.
        tp, tn, fp, fn = get_confusion_matrix(outputs, labels, threshold=0.5)
        epoch_tp += tp
        epoch_tn += tn
        epoch_fp += fp
        epoch_fn += fn

        if scheduler_mode == 'batch':
            scheduler.step()

    if scheduler_mode == 'epoch':
        scheduler.step()

    epoch_loss /= (batch_idx + 1)

    return epoch_loss, epoch_tp, epoch_tn, epoch_fp, epoch_fn


@torch.inference_mode()
def _validate_one_epoch(model, dataloader, criterion, use_amp, rank=None):
    if rank is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cuda', rank)  # distributed data parallel

    model.eval()
    valid_loss, tp_valid, tn_valid, fp_valid, fn_valid = 0, 0, 0, 0, 0

    if rank is None or rank == 0:
        dataloader = tqdm(dataloader, desc=f'Validation')  # the processor other than rank0 do not show progressbar.

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # BCEWithLogitsLoss, binary cross entropy from_logits
            outputs = F.sigmoid(outputs)

        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()

        valid_loss += loss.item()
        tp, tn, fp, fn = get_confusion_matrix(outputs, labels, threshold=0.5)
        tp_valid += tp
        tn_valid += tn
        fp_valid += fp
        fn_valid += fn

    valid_loss /= (batch_idx + 1)

    return valid_loss, tp_valid, tn_valid, fp_valid, fn_valid


def train_DDP(rank, world_size, config):
    model1 = config.model
    ds_train = config.ds_train
    ds_valid = config.ds_valid
    criterion = config.criterion
    epochs_num = config.epochs_num
    optimizer = config.optimizer
    scheduler = config.scheduler
    scheduler_mode = config.scheduler_mode
    use_amp = config.use_amp
    batch_size = config.batch_size
    num_workers = config.num_workers
    save_model_dir = config.save_model_dir
    save_model_dir.mkdir(parents=True, exist_ok=True)
    losses_pkl = config.losses_pkl

    setup(rank, world_size)

    torch.cuda.set_device(rank)  # dist.all_gather_object runs much quicker.
    if config.sync_bn:
        model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)
    model_ddp = DDP(model1, device_ids=[rank])
    sampler_train = DistributedSampler(ds_train, world_size, rank)
    loader_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler_train, num_workers=num_workers, pin_memory=True)

    if rank == 0:
        logging.basicConfig(filename=save_model_dir / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)
        list_loss_history = []  # using list_loss_history to draw the line of training and validation loss
    for epoch in range(epochs_num):
        sampler_train.set_epoch(epoch)
        loss_train, tp_train, tn_train, fp_train, fn_train = \
            _train_one_epoch(model_ddp, loader_train, criterion, epoch, optimizer, scheduler, scheduler_mode, use_amp, rank=rank)
        dist.barrier()
        data = {
            'loss': loss_train, 'tp': tp_train, 'tn': tn_train, 'fp': fp_train, 'fn': fn_train,
        }
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, data)

        if rank == 0:
            list_loss = []
            for index, output in enumerate(outputs):
                list_loss.append(output['loss'])
                if index == 0:
                    tp, tn, fp, fn = 0, 0, 0, 0
                else:
                    tp += output['tp']
                    tn += output['tn']
                    fp += output['fp']
                    tn += output['tn']

            loss_train_avg = np.mean(list_loss)  #combine loss from multiple processes.
            print(f'training {epoch} loss:{loss_train_avg:.3f}')
            acc_train, sen_train, spe_train, iou_train, dice_train = get_metrics(tp, tn, fp, fn)
            print(f'training epoch{epoch} metrics:')
            print(f'losses:{loss_train:.3f}')
            print(f'acc:{acc_train:.3f}, sen:{sen_train:.3f}, spe:{spe_train:.3f}')
            print(f'iou:{iou_train:.3f}, dice:{dice_train:.3f}')

        print(f'epoch:{epoch} compute validation dataset...')
        dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        loss_valid, tp_valid, tn_valid, fp_valid, fn_valid = _validate_one_epoch(model_ddp, dataloader_valid, criterion, use_amp=use_amp, rank=rank)
        dist.barrier()
        data = {
            'loss': loss_valid, 'tp': tp_valid, 'tn': tn_valid, 'fp': fp_valid, 'fn': fn_valid,
        }
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, data)

        if rank == 0:
            list_loss = []
            tp, tn, fp, fn = 0, 0, 0, 0
            for output in outputs:
                list_loss.append(output['loss'])
                tp += output['tp']
                tn += output['tn']
                fp += output['fp']
                tn += output['tn']

            loss_valid_avg = np.mean(list_loss)  # combine loss from multiple processes.
            acc_valid, sen_valid, spe_valid, dice_valid = get_metrics(tp, tn, fp, fn)
            print(f'validation metrics:')
            print(f'losses:{loss_valid_avg:.3f}')
            print(f'acc:{acc_valid:.3f}, sen:{sen_valid:.3f}, spe:{spe_valid:.3f}')
            print(f'dice:{dice_valid:.3f}')

            list_loss_history.append([float(loss_train_avg), float(loss_valid_avg)])
            save_model_file = save_model_dir / f'valid_loss_{loss_valid_avg:.3f)}_epoch{epoch}.pth'
            try:
                state_dict = model1.module.state_dict()
            except AttributeError:
                state_dict = model1.state_dict()
            print('save model:', save_model_file)
            torch.save(state_dict, save_model_file)

    if rank == 0:
        pickle.dump(list_loss_history, open(losses_pkl, 'wb'))
    cleanup()
    clear_gpu_cache()



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache():
    if torch.cuda.device_count() > 0:
        gc.collect()
        torch.cuda.empty_cache()


def draw_loss_graph(list_losses, save_img_file=None):
    # train_losses = [loss[0] for loss in list_losses]
    # val_losses = [loss[1] for loss in list_losses]

    (train_losses, val_losses) = tuple(zip(*list_losses))

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_img_file is not None:
        plt.savefig(save_img_file, bbox_inches='tight') #save image file should be executed before calling plt.show()
    else:
        plt.show()
        # plt.close()


#endregion