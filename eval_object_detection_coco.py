import numpy as np
import glob
from torchvision.transforms import functional as F
from tqdm import tqdm
import torch
from vision.torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from vision.torchvision.models.detection.ssd import ssd300_vgg16
import os
import argparse
import presets
from coco.coco_utils import get_coco, get_coco_api_from_dataset
from coco.coco_eval import CocoEvaluator
from torchvision.utils import save_image
from pytorch_faster_rcnn import PyTorchFasterRCNN
from attack.pgd_patch import PGDPatch
from attack.mim_patch import MIMPatch
from patch_detector import PatchDetector
from typing import Tuple, Dict, Optional
from dice_loss import dice_coeff, precision, recall
from utils import save_detection_image
from attack.utils import create_triangle_mask, create_diamond_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


class ToTensor(torch.nn.Module):
    def forward(self, image: torch.Tensor,
                target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.to_tensor(image)
        return image, target


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


parser = argparse.ArgumentParser(description="Evaluate object detector")
parser.add_argument('--n_batches', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=100)
parser.add_argument('--patch_height', type=int, default=100)
parser.add_argument('--patch_width', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--skip_benign', action='store_true', default=False)
parser.add_argument('--skip_attack', action='store_true', default=False)
parser.add_argument('--eval_dir', type=str, default='runs/test')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--eval_round', type=int, default=0)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--vis_dir', type=str, default='eval_coco')
parser.add_argument('--defense', action='store_true', default=False)
parser.add_argument('--n_filter', type=int, default=16)
parser.add_argument('--adaptive', action='store_true', default=False)
parser.add_argument('--adaptive_to_shape_completion', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--model', type=str, default='faster_rcnn')
parser.add_argument('--triangle', action='store_true', default=False)
parser.add_argument('--diamond', action='store_true', default=False)
parser.add_argument('--use_label', action='store_true', default=False)
parser.add_argument('--bpda', action='store_true', default=False)  # use sigmoid patch mask instead of 0-1 mask in order for gradient propagation during attack
parser.add_argument('--shape_completion', action='store_true', default=False)  # post-processing during inference to complete the square
parser.add_argument('--simple_shape_completion', action='store_true', default=False)
parser.add_argument('--circle', action='store_true', default=False)
parser.add_argument('--rectangle', action='store_true', default=False)
parser.add_argument('--ellipse', action='store_true', default=False)
parser.add_argument('--bpda_shape_completion', action='store_true', default=False)
parser.add_argument('--union', action='store_true', default=False)
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument('--load_mask', action='store_true', default=False, help='load fixed patch mask')
parser.add_argument('--attack', type=str, default='pgd', help='attack method')
parser.add_argument('--n_patch', type=int, default=1)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "coco"
dataset = get_coco(data_path, "val", transforms=presets.DetectionPresetEval())
dataset = torch.utils.data.Subset(dataset, list(range(args.n_batches)))

data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn)
patch_size = args.patch_size
if args.model == 'faster_rcnn':
    model = fasterrcnn_resnet50_fpn(pretrained=True)
elif args.model == 'ssd':
    print("Use ssd")
    model = ssd300_vgg16(pretrained=True)

if args.weights_path is not None:
    model.load_state_dict(torch.load(args.weights_path))
model.eval()
model.cuda()

detector = PatchDetector(3, 1, base_filter=args.n_filter, square_sizes=[125, 100, 75, 50, 25], n_patch=args.n_patch)
eval_dir = args.eval_dir
if args.ckpt:
    detector.unet.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
detector.cuda()
detector.eval()



art_model = PyTorchFasterRCNN(
        model=model,
        detector=detector,
        clip_values=(0, 1.0),
        channels_first=False,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type=DEVICE,
        adaptive=args.adaptive,
        defense=args.defense,
        bpda=args.bpda,
        shape_completion=args.shape_completion,
        adaptive_to_shape_completion=args.adaptive_to_shape_completion,
        simple_shape_completion=args.simple_shape_completion,
        bpda_shape_completion=args.bpda_shape_completion,
        union=args.union
    )


art_model_undefended = PyTorchFasterRCNN(
        model=model,
        detector=detector,
        clip_values=(0, 1.0),
        channels_first=False,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type=DEVICE,
        adaptive=False,
        defense=False,
        bpda=False,
        shape_completion=False,
        adaptive_to_shape_completion=False,
        simple_shape_completion=False,
        bpda_shape_completion=False
    )

for param in model.parameters():
    param.requires_grad = False
if args.rectangle or args.ellipse:
    patch_height = args.patch_height
    patch_width = args.patch_width
else:
    patch_height = args.patch_size
    patch_width = args.patch_size

if args.rectangle or args.ellipse:
    data_dir = f'{args.data_dir}/{patch_height}x{patch_width}'
elif args.load_mask:
    data_dir = args.data_dir
elif args.triangle:
    data_dir = f'{args.data_dir}/triangle_{args.patch_size}'
elif args.diamond:
    data_dir = f'{args.data_dir}/diamond_{args.patch_size}'
else:
    data_dir = f'{args.data_dir}/{args.patch_size}'

data_files = sorted(glob.glob(os.path.join(data_dir, "*pth.tar")))

if args.vis:
    if args.rectangle or args.ellipse:
        vis_dir = os.path.join(args.eval_dir, f'round_{args.eval_round}/patch_size-{patch_height}x{patch_width}/vis')
    elif args.load_mask:
        vis_dir = os.path.join(args.eval_dir, 'vis')
    else:
        vis_dir = os.path.join(args.eval_dir, f'round_{args.eval_round}/patch_size-{args.patch_size}/vis')
    os.makedirs(vis_dir, exist_ok=True)

if args.save_results:
    if args.rectangle or args.ellipse:
        result_dir = os.path.join(args.eval_dir, f'round_{args.eval_round}/patch_size-{patch_height}x{patch_width}/results')
    elif args.load_mask:
        result_dir = os.path.join(args.eval_dir, 'results')
    else:
        result_dir = os.path.join(args.eval_dir, f'round_{args.eval_round}/patch_size-{args.patch_size}/results')
    os.makedirs(result_dir, exist_ok=True)



if args.skip_benign:
    print("Skipping inference on benign examples...")
else:
    print("Running inference on benign examples...")
    dice_score_benign = 0
    precision_benign = 0
    recall_benign = 0
    coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(data_loader.dataset), ['bbox'])

    for batch_idx, data_file in enumerate(tqdm(data_files[0:args.n_batches], desc="Benign")):
        data = torch.load(data_file)
        x = data['x']
        y = data['y']
        x = x[0].permute(1, 2, 0).unsqueeze(0).numpy()  # 1, h, w, 3

        y_pred, x_processed, mask, raw_mask = art_model.predict(x)

        y_pred = [{k: torch.from_numpy(v) for k, v in t.items()} for t in y_pred]
        res = {target["image_id"].item(): output for target, output in zip(y, y_pred)}
        coco_evaluator.update(res)

        if args.defense:
            true_mask = torch.zeros_like(mask[0])
            dice_score_benign += dice_coeff(mask[0], true_mask.cuda()).item()
            precision_benign += precision(mask[0], true_mask.cuda()).item()
            recall_benign += recall(mask[0], true_mask.cuda()).item()
        if args.vis:
            x_processed = x_processed[0].unsqueeze(0)
            save_image(x_processed, os.path.join(vis_dir, f'{batch_idx}_clean_masked.png'))

    dice_score_benign = dice_score_benign / args.n_batches
    precision_benign = precision_benign / args.n_batches
    recall_benign = recall_benign / args.n_batches
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

if args.skip_attack:
    print("Skipping attack generation...")
else:
    print("Running inference on adversarial examples...")
    coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(data_loader.dataset), ['bbox'])
    dice_score_adv = 0
    precision_adv = 0
    recall_adv = 0
    if args.attack == 'pgd':
        attacker = PGDPatch(art_model, batch_size=args.batch_size, eps=args.eps,
                        eps_step=2*args.eps/args.max_iter, max_iter=args.max_iter, num_random_init=0, random_eps=False, targeted=False, verbose=True)
    elif args.attack == 'mim':
        attacker = MIMPatch(art_model, batch_size=args.batch_size, eps=args.eps,
                        eps_step=2*args.eps/args.max_iter, max_iter=args.max_iter, num_random_init=0, random_eps=False, targeted=False, verbose=True)
    for batch_idx, data_file in enumerate(tqdm(data_files[0:args.n_batches], desc="Attack")):
        if args.save_results:
            result_fn = os.path.join(result_dir, f'adv_results_{batch_idx}.pth.tar')
            is_done = os.path.exists(result_fn)
        else:
            is_done = False
        if not is_done:
            data = torch.load(data_file)
            x = data['x']
            y = data['y']
            x = x[0].permute(1, 2, 0).unsqueeze(0).numpy()
            if args.use_label:
                label = [{k: v.numpy() for k, v in t.items()} for t in y]
                label = label[0]
            else:
                label = None
            if args.load_mask:
                patch_mask = data['mask']
                x_adv = attacker.generate(x, y=label, mask=patch_mask)
            else:
                xmin, ymin = data['idx'][args.eval_round, :2]
                xmin = int(xmin)
                ymin = int(ymin)
                if args.circle:
                    x_adv = attacker.generate(x, y=label, d=args.patch_size, xmin=int(xmin), ymin=int(ymin))
                elif args.triangle:
                    patch_mask = create_triangle_mask(x.shape[1:], xmin, ymin, args.patch_size)
                    x_adv = attacker.generate(x, y=label, mask=patch_mask)
                elif args.diamond:
                    patch_mask = create_diamond_mask(x.shape[1:], xmin, ymin, args.patch_size)
                    x_adv = attacker.generate(x, y=label, mask=patch_mask)
                elif args.ellipse:
                    x_adv = attacker.generate(x, y=label, a=patch_width/2, b=patch_height/2, xmin=int(xmin), ymin=int(ymin))
                else:
                    x_adv = attacker.generate(x, y=label, patch_height=patch_height, patch_width=patch_width,
                                              xmin=int(xmin), ymin=int(ymin))
            y_pred, x_processed, mask, raw_mask = art_model.predict(x_adv)
            y_pred_np = y_pred
            y_pred = [{k: torch.from_numpy(v) for k, v in t.items()} for t in y_pred]

            if args.defense:
                if args.load_mask or args.triangle or args.diamond:
                    true_mask = torch.from_numpy(patch_mask[:, :, 0]).unsqueeze(0).unsqueeze(0).float()
                else:
                    if args.circle:
                        h = x.shape[1]
                        w = x.shape[2]
                        Y, X = np.ogrid[:h, :w]
                        radius = args.patch_size / 2
                        xcenter = xmin + radius
                        ycenter = ymin + radius
                        dist_from_center = np.sqrt((X - xcenter) ** 2 + (Y - ycenter) ** 2)
                        circle_mask = dist_from_center <= radius
                        # pdb.set_trace()
                        true_mask = torch.from_numpy(circle_mask).unsqueeze(0).unsqueeze(0).float()
                    elif args.ellipse:
                        h = x.shape[1]
                        w = x.shape[2]
                        Y, X = np.ogrid[:h, :w]
                        a = patch_width/2
                        b = patch_height/2
                        xcenter = xmin + a
                        ycenter = ymin + b
                        dist_from_center = (X - xcenter) ** 2 / a**2 + (Y - ycenter) ** 2/b**2
                        circle_mask = dist_from_center <= 1
                        true_mask = torch.from_numpy(circle_mask).unsqueeze(0).unsqueeze(0).float()
                    else:
                        true_mask = torch.zeros_like(mask[0])
                        true_mask[:, :, ymin:ymin+patch_height, xmin:xmin+patch_width] = 1
                dice_score_adv += dice_coeff(mask[0], true_mask.cuda()).item()
                precision_adv += precision(mask[0], true_mask.cuda()).item()
                recall_adv += recall(mask[0], true_mask.cuda()).item()
                if args.vis:
                    save_image(true_mask, os.path.join(vis_dir, f'{batch_idx}_adv_mask_gt.png'))
                    save_image(mask[0], os.path.join(vis_dir, f'{batch_idx}_adv_mask_pred.png'))
                    save_image(raw_mask[0], os.path.join(vis_dir, f'{batch_idx}_adv_raw_mask_pred.png'))
            if args.vis:
                x_processed = x_processed[0].unsqueeze(0)
                save_image(x_processed, os.path.join(vis_dir, f'{batch_idx}_adv_masked.png'))
                save_image(F.to_tensor(x_adv[0]).unsqueeze(0), os.path.join(vis_dir, f'{batch_idx}_adv.png'))
                
                x_processed_adv = x_processed.permute(0, 2, 3, 1).cpu().numpy()
                y_pred_undefended, _, _, _ = art_model_undefended.predict(x_adv)
                save_detection_image(x_processed_adv, y_pred_np, os.path.join(vis_dir, f'{batch_idx}_pred_defended.png'))
                save_detection_image(x_adv, y_pred_undefended, os.path.join(vis_dir, f'{batch_idx}_pred_undefended.png'))
            if args.save_results:
                if args.load_mask:
                    data['y_pred'] = y_pred
                    data['x_adv'] = x_adv
                    torch.save(data, result_fn)
                else:
                    torch.save({'y': y, 'y_pred': y_pred, 'x_adv': x_adv, 'x': x, 'xmin': xmin, 'ymin':ymin}, result_fn)
        else:
            result = torch.load(result_fn)
            y = result['y']
            y_pred = result['y_pred']
            if type(y_pred[0]['labels']) == np.ndarray:
                y_pred = [{k: torch.from_numpy(v) for k, v in t.items()} for t in y_pred]
        res = {target["image_id"].item(): output for target, output in zip(y, y_pred)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    dice_score_adv = dice_score_adv / args.n_batches
    precision_adv = precision_adv / args.n_batches
    recall_adv = recall_adv / args.n_batches

if not args.skip_attack:
    print("********** Adv Segmentation Scores ***************")
    print(f'Dice Adv: {dice_score_adv}')
    print(f'Precision Adv: {precision_adv}')
    print(f'Recall Adv: {recall_adv}')
if not args.skip_benign:
    print("********** Benign Segmentation Scores ***************")
    print(f'Dice Benign: {dice_score_benign}')
    print(f'Precision Benign: {precision_benign}')
    print(f'Recall Benign: {recall_benign}')
