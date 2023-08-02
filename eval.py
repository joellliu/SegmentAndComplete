import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dice_loss import dice_coeff
from torchvision.utils import save_image
import os

def eval_net(net, loader, device, epoch=0, global_step=0, dir_vis='vis'):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    save_idx = np.random.randint(0, n_val)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_idx, (imgs, true_masks) in enumerate(loader):
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()
            if batch_idx == save_idx:
                save_image(imgs, os.path.join(dir_vis, f'eval_images_{epoch}_{global_step}_{save_idx}.png'))
                save_image(mask_pred, os.path.join(dir_vis, f'eval_masks_pred_{epoch}_{global_step}_{save_idx}.png'))
                save_image(true_masks, os.path.join(dir_vis, f'eval_masks_gt_{epoch}_{global_step}_{save_idx}.png'))
    net.train()
    return tot / n_val
