#!/bin/sh
device=$1
patch_size=$2
eval_dir="coco_at_bpda_sc_bpda"
ckpt="ckpts/coco_at.pth"

mkdir evaluation/$eval_dir
for round in 0 1 2
do
    CUDA_VISIBLE_DEVICES=$device python eval_object_detection_coco.py --eval_dir evaluation/$eval_dir --eval_round $round --vis --patch_size $patch_size --ckpt $ckpt --use_label --defense --adaptive --bpda --shape_completion --bpda_shape_completion | tee -a "evaluation/$eval_dir/round-$round-patch_size-$patch_size.txt"
done
