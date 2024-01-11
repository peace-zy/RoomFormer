#!/usr/bin/env bash

python infer.py --filename="imgs/d.jpg" \
               --checkpoint=/mnt/aigc_ssd/zhangyan461/models/roomformer/checkpoints/roomformer_stru3d_semantic_rich.pth \
               --output_dir=infer_stru3d_sem_rich \
               --num_queries=2800 \
               --num_polys=70 \
               --semantic_classes=19
