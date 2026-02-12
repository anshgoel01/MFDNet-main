import os
from config import Config
import torch
torch.backends.cudnn.benchmark = True

from SSIM import SSIM
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
from MFDNet import HPCNet as mfdnet
import losses
from tqdm import tqdm

torch.set_num_threads(os.cpu_count())


if __name__ == "__main__":

    opt = Config('training.yml')

    print("TRAIN_DIR =", opt.TRAINING.TRAIN_DIR)
    print("VAL_DIR   =", opt.TRAINING.VAL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    file_psnr = 'MFD_PSNR.txt'
    file_loss = 'MFD_LOSS.txt'

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    # -------------------------------------------------------
    # Data
    # -------------------------------------------------------
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    model_restoration = mfdnet().to(device)

    optimizer = optim.Adam(
        model_restoration.parameters(),
        lr=opt.OPTIM.LR_INITIAL,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)

    criterion_char = losses.CharbonnierLoss().to(device)
    criterion_edge = losses.EdgeLoss().to(device)
    criterion_ssim = SSIM().to(device)

    print(f"\nStart Epoch {start_epoch} → End Epoch {opt.OPTIM.NUM_EPOCHS}\n")

    best_psnr = 0
    best_epoch = 0

    # =======================================================
    # TRAINING LOOP
    # =======================================================
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):

        epoch_start_time = time.time()
        epoch_loss = 0
        SSIM_all = 0
        train_sample = 0

        model_restoration.train()

        for i, data in enumerate(train_loader):

            print(f"\nBatch {i} entered loop")

            optimizer.zero_grad()

            target = data[0].to(device)
            input_ = data[1].to(device)

            print("Input shape :", input_.shape)
            print("Target shape:", target.shape)

            print("Forward pass starting...")

            restored = model_restoration(input_)

            print("Forward pass done")

            # ✅ SAFE OUTPUT HANDLING
            if isinstance(restored, (list, tuple)):
                restored_stage1 = restored[0]
                restored_stage2 = restored[1]
            else:
                restored_stage1 = restored
                restored_stage2 = restored

            loss_char0 = criterion_char(restored_stage1, target)
            loss_char1 = criterion_char(restored_stage2, input_)

            loss_edge0 = criterion_edge(restored_stage1, target)

            loss_ssim0 = criterion_ssim(restored_stage1, target)
            loss_ssim1 = criterion_ssim(restored_stage2, input_)

            loss = (
                0.3 * (loss_char0 + 0.2 * loss_char1)
                + 0.2 * loss_edge0
                - 0.15 * (loss_ssim0 + 0.2 * loss_ssim1)
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            SSIM_all += loss_ssim0.item()
            train_sample += 1

        avg_ssim = SSIM_all / train_sample

        print("\n------------------------------------------------")
        print(f"Epoch {epoch} | Time {time.time()-epoch_start_time:.2f}s")
        print(f"Loss {epoch_loss:.4f} | SSIM {avg_ssim:.4f}")
        print("------------------------------------------------\n")

        scheduler.step()
