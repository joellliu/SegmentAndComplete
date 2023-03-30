import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset import COCODataset
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image



def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              patch_size=100,
              image_size=500,
              dir='runs/0501/',
              p_clean=0.3,
              data_path="../FasterRCNN_adv_dataset/adv_dataset_random_patch_100/data/"):
    dir_checkpoint = os.path.join(dir, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_vis = os.path.join(dir, 'vis')
    os.makedirs(dir_vis, exist_ok=True)

    dataset = COCODataset(data_path, output_size=image_size, patch_size=patch_size, p_clean=p_clean, random_crop=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, _ = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(51))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    dataset_clean = COCODataset(data_path, output_size=image_size, patch_size=patch_size, p_clean=1, random_crop=False)
    _, val_clean = random_split(dataset_clean, [n_train, n_val], generator=torch.Generator().manual_seed(51))
    val_loader_clean = DataLoader(val_clean, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    dataset_adv = COCODataset(data_path, output_size=image_size, patch_size=patch_size, p_clean=0, random_crop=False)
    _, val_adv = random_split(dataset_adv, [n_train, n_val], generator=torch.Generator().manual_seed(51))
    val_loader_adv = DataLoader(val_adv, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=os.path.join(dir, 'logs'), comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Patch Size:  {patch_size}
        Image Size: {image_size}
        Clean ratio: {p_clean*100}%
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (5 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score_clean = eval_net(net, val_loader_clean, device, epoch, global_step, dir_vis=dir_vis)
                    val_score_adv = eval_net(net, val_loader_adv, device, epoch, global_step, dir_vis=dir_vis)
                    scheduler.step(val_score_clean*p_clean + (1-p_clean)*val_score_adv)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy adv: {}'.format(val_score_adv))
                        logging.info('Validation cross entropy clean: {}'.format(val_score_clean))
                        writer.add_scalar('Loss/test_clean', val_score_clean, global_step)
                        writer.add_scalar('Loss/test_adv', val_score_adv, global_step)
                    else:
                        logging.info('Validation Dice Coeff adv: {}'.format(val_score_adv))
                        logging.info('Validation Dice Coeff clean: {}'.format(val_score_clean))
                        writer.add_scalar('Dice/test_clean', val_score_clean, global_step)
                        writer.add_scalar('Dice/test_adv', val_score_adv, global_step)

                    torch.save(net.state_dict(),
                               os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}_{global_step}.pth'))
                    logging.info(f'Checkpoint {epoch + 1} {global_step} saved !')
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                        save_image(imgs, os.path.join(dir_vis, f'images_{global_step}.png'))
                        save_image(masks_pred, os.path.join(dir_vis, f'masks_pred_{global_step}.png'))
                        save_image(true_masks, os.path.join(dir_vis, f'masks_gt_{global_step}.png'))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--p_clean', dest='p_clean', type=float, default=0.3,
                        help='Ratio of the data that is used as validation (0-1)')
    parser.add_argument('-n', '--n_filter', dest='n_filter', type=int, default=64,
                        help='Number of base filters for unet')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--device', dest='device', default='0', type=str, help='GPU to use.')
    parser.add_argument('--data_path', type=str, default="adv_data/coco_topleft_patch_100/data/")
    parser.add_argument('-dir', '--dir', dest='dir', default='runs/0504/', type=str, help='dir for saving files')
    parser.add_argument('-s', '--image_size', dest='image_size', type=int, default=500,
                        help='Image size for training')
    parser.add_argument('-p', '--patch_size', dest='patch_size', type=int, default=100,
                        help='Patch size in the original image')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.dir, exist_ok=True)
    log_file = os.path.join(args.dir, 'logging.txt')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=log_file,
                        filemode='w' if not args.load else 'a')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True, base_filter=args.n_filter)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
                 f'\t{args.n_filter} base filters'
                 f'\t{args.p_clean*100}% clean data')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  patch_size=args.patch_size,
                  image_size=args.image_size,
                  dir=args.dir,
                  data_path=args.data_path,
                  p_clean=args.p_clean
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
