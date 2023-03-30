import argparse
import logging
import os
import sys
from dice_loss import dice_coeff
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
import warnings
from unet import UNet
import torch.backends.cudnn as cudnn
from dataset import COCODatasetATPD
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torch.distributed as dist
from advertorch.context import ctx_noparamgrad_and_eval
import torch.multiprocessing as mp

dir_data = "adv_data/coco_topleft_patch_100/data/"


def mask_pgd_attack(net, x_natural, mask, gpu, criterion=nn.BCEWithLogitsLoss(), eps=1.0, max_iter=200, eps_step=0.01,
                    random_init=False):
    with ctx_noparamgrad_and_eval(net):
        x_adv = x_natural.detach()
        if random_init:
            noise = 0.001*torch.randn(x_natural.shape).to(gpu)
            noise = torch.where(mask == 0.0, torch.tensor(0.0).to(gpu), noise)
            x_adv = x_adv + noise.detach()
        for _ in range(max_iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = criterion(net(x_adv), mask)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            # mask the gradient outside the patch
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(gpu), grad)
            grad = grad.sign()
            x_adv = x_adv.detach() + eps_step * grad.detach()
            x_adv = torch.min(torch.max(x_adv, x_natural - eps), x_natural + eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def eval_net(net, loader, args, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    dir = args.dir
    dir_vis = os.path.join(dir, 'vis')

    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    tot_adv = 0
    with ctx_noparamgrad_and_eval(net):
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for imgs, masks_clean, masks_adv in loader:
                masks_clean = masks_clean.to(args.gpu, dtype=mask_type)
                masks_adv = masks_adv.to(args.gpu, dtype=mask_type)
                imgs = imgs.to(args.gpu, dtype=torch.float32)
                imgs_adv = mask_pgd_attack(net, imgs, masks_adv, args.gpu)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    mask_pred_adv = net(imgs_adv)

                pred = torch.sigmoid(mask_pred)
                pred_adv = torch.sigmoid(mask_pred_adv)
                pred = (pred > 0.5).float()
                pred_adv = (pred_adv > 0.5).float()
                tot += dice_coeff(pred, masks_clean).item()
                tot_adv += dice_coeff(pred_adv, masks_adv).item()
                pbar.update(imgs.shape[0])

    tot = tot / n_val
    tot_adv = tot_adv / n_val
    if args.multiprocessing_distributed:
        tot_tensor = torch.tensor(tot).to(args.gpu)
        tot_adv_tensor = torch.tensor(tot_adv).to(args.gpu)
        dist.all_reduce(tot_adv_tensor, dist.ReduceOp.SUM)
        dist.all_reduce(tot_tensor, dist.ReduceOp.SUM)
        tot = tot_tensor.cpu().numpy() / args.world_size
        tot_adv = tot_adv_tensor.cpu().numpy() / args.world_size
        print(f"Rank {args.rank} Acc {tot} {tot_adv}")

    if args.main_process:
        save_image(imgs, os.path.join(dir_vis, f'eval_images_clean_{epoch + 1}.png'))
        save_image(imgs_adv, os.path.join(dir_vis, f'eval_images_adv_{epoch + 1}.png'))
        save_image(pred, os.path.join(dir_vis, f'eval_masks_pred_clean_{epoch + 1}.png'))
        save_image(pred_adv, os.path.join(dir_vis, f'eval_masks_pred_adv_{epoch + 1}.png'))
        save_image(masks_clean, os.path.join(dir_vis, f'eval_masks_gt_clean_{epoch + 1}.png'))
        save_image(masks_adv, os.path.join(dir_vis, f'eval_masks_gt_adv_{epoch + 1}.png'))

    return tot_adv, tot



def train_net(net,
              args,
              ngpus_per_node,
              save_cp=True):
    epochs = args.epochs
    batch_size = args.batch_size
    print(f'batch size {batch_size}')
    lr = args.lr
    val_percent = args.val / 100
    patch_size = args.patch_size
    image_size = args.image_size
    dir = args.dir
    p_clean = args.p_clean

    dir_checkpoint = os.path.join(dir, 'checkpoints')
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_vis = os.path.join(dir, 'vis')
    os.makedirs(dir_vis, exist_ok=True)

    # set dataset, dataloader
    dataset = COCODatasetATPD(dir_data, output_size=image_size, patch_size=patch_size, random_crop=True)
    dataset_val = COCODatasetATPD(dir_data, output_size=image_size, patch_size=patch_size, random_crop=False)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, _ = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(51))
    _, val = random_split(dataset_val, [n_train, n_val], generator=torch.Generator().manual_seed(51))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            sampler=test_sampler)


    if args.main_process:
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            GPU:          {args.gpu}
            Patch Size:  {patch_size}
            Image Size: {image_size}
            Clean ration: {p_clean * 100}%
        ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*epochs), int(0.8*epochs)], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', disable=not args.main_process) as pbar:
            global_step = 0
            for imgs, masks_clean, masks_adv in train_loader:
                imgs = imgs.to(args.gpu, dtype=torch.float32)
                # masks_clean = masks_clean.to(args.gpu, dtype=torch.float32)
                # masks_adv = masks_adv.to(args.gpu, dtype=torch.float32)
                # imgs_adv = mask_pgd_attack(net, imgs, masks_adv, gpu=args.gpu, criterion=criterion)

                if np.random.uniform() < p_clean:
                    true_masks = masks_clean.to(args.gpu, dtype=torch.float32)
                else:
                    true_masks = masks_adv.to(args.gpu, dtype=torch.float32)
                    imgs = mask_pgd_attack(net, imgs, true_masks, gpu=args.gpu, criterion=criterion)

                masks_pred = net(imgs)
                # masks_pred_adv = net(imgs_adv)
                loss = criterion(masks_pred, true_masks)
                # loss = p_clean*criterion(masks_pred, masks_clean) + (1-p_clean)*criterion(masks_pred_adv, masks_adv)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0]*ngpus_per_node)
                global_step += 1
                if args.main_process and global_step % 10 == 0:
                    save_image(imgs, os.path.join(dir_vis, f'images_{epoch + 1}_{global_step}.png'))
                    # save_image(imgs_adv, os.path.join(dir_vis, f'images_adv_{epoch + 1}_{global_step}.png'))
                    save_image(masks_pred, os.path.join(dir_vis, f'masks_pred_{epoch + 1}_{global_step}.png'))
                    # save_image(masks_pred_adv, os.path.join(dir_vis, f'masks_pred_adv_{epoch + 1}_{global_step}.png'))
                    save_image(true_masks, os.path.join(dir_vis, f'masks_gt_{epoch + 1}_{global_step}.png'))
                    # save_image(masks_adv, os.path.join(dir_vis, f'masks_gt_adv_{epoch + 1}_{global_step}.png'))
                if global_step % args.val_freq == 0:
                    val_score_adv, val_score_clean = eval_net(net, val_loader, args, epoch)
                    if args.main_process:
                        logging.info('Epoch {} Step {} Validation Dice Coeff adv: {}'.format(epoch, global_step,
                                                                                             val_score_adv))
                        logging.info('Epoch {} Step {} Validation Dice Coeff clean: {}'.format(epoch, global_step,
                                                                                               val_score_clean))
                        if save_cp:
                            torch.save(net.module.state_dict() if args.distributed else net.state_dict(),
                                       os.path.join(dir_checkpoint, f'CP_epoch_{epoch + 1}_{global_step}.pth'))
                            logging.info(f'Checkpoint {epoch + 1}_{global_step}saved !')

        val_score_adv, val_score_clean = eval_net(net, val_loader, args, epoch)
        scheduler.step()

        if args.main_process:
            logging.info('Validation Dice Coeff adv: {}'.format(val_score_adv))
            logging.info('Validation Dice Coeff clean: {}'.format(val_score_clean))

            save_image(imgs, os.path.join(dir_vis, f'images_{epoch + 1}_{global_step}.png'))
            # save_image(imgs_adv, os.path.join(dir_vis, f'images_adv_{epoch + 1}_{global_step}.png'))
            save_image(masks_pred, os.path.join(dir_vis, f'masks_pred_{epoch + 1}_{global_step}.png'))
            # save_image(masks_pred_adv, os.path.join(dir_vis, f'masks_pred_adv_{epoch + 1}_{global_step}.png'))
            save_image(true_masks, os.path.join(dir_vis, f'masks_gt_{epoch + 1}_{global_step}.png'))
            # save_image(masks_adv, os.path.join(dir_vis, f'masks_gt_adv_{epoch + 1}_{global_step}.png'))

            if save_cp:
                torch.save(net.module.state_dict() if args.distributed else net.state_dict(),
                           os.path.join(dir_checkpoint, f'CP_epoch_{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=81,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-r', '--p_clean', dest='p_clean', type=float, default=0.3,
                        help='Ratio of the data that is used as validation (0-1)')
    parser.add_argument('-n', '--n_filter', dest='n_filter', type=int, default=64,
                        help='Number of base filters for unet')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('-d', '--device', dest='device', default='0', type=str, help='GPU to use.')
    parser.add_argument('-dir', '--dir', dest='dir', default='runs/0511-at/', type=str, help='dir for saving files')
    parser.add_argument('-s', '--image_size', dest='image_size', type=int, default=500,
                        help='Image size for training')
    parser.add_argument('-p', '--patch_size', dest='patch_size', type=int, default=100,
                        help='Patch size in the original image')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://localhost:8888', help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    return parser.parse_args()


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    os.makedirs(args.dir, exist_ok=True)


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True, base_filter=args.n_filter)
    args.main_process = not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0)
    if args.main_process:
        log_file = os.path.join(args.dir, f'logging.txt')
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=log_file,
                            filemode='w' if not args.load else 'a')
        logging.info(f'Network:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
                     f'\t{args.n_filter} base filters'
                     f'\t{args.p_clean * 100}% clean data')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location='cpu')
        )
        logging.info(f'Model loaded from {args.load}')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net.cuda(args.gpu)
            net_distributed = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                                        find_unused_parameters=True,
                                                                        broadcast_buffers=False)
        else:
            net.cuda()
            net_distributed = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net_distributed = net.cuda(args.gpu)
    else:
        net_distributed = torch.nn.DataParallel(net).cuda()

    # faster convolutions, but more memory
    cudnn.benchmark = True

    try:
        train_net(net=net_distributed,
                  args=args,
                  ngpus_per_node=ngpus_per_node
                  )
    except KeyboardInterrupt:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


def main():
    args = get_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
