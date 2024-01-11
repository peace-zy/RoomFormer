import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
from shapely.geometry import Polygon

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from engine import evaluate_floor
from models import build_model
from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
from s3d_floorplan_eval.options import MCSSOptions
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan

def get_args_parser():
    parser = argparse.ArgumentParser('RoomFormer', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots (num_polys * max. number of corner per poly)")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Number of maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")
    parser.add_argument('--masked_attn', default=False, action='store_true',
                        help="if true, the query in one room will not be allowed to attend other room")
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty")

    # aux
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # dataset parameters
    parser.add_argument('--dataset_name', default='stru3d')
    parser.add_argument('--dataset_root', default='data/stru3d', type=str)
    parser.add_argument('--eval_set', default='test', type=str)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/roomformer_scenecad.pth', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='eval_stru3d',
                        help='path where to save result')

    # visualization options
    parser.add_argument('--plot_pred', default=True, type=bool, help="plot predicted floorplan")
    parser.add_argument('--plot_density', default=True, type=bool, help="plot predicited room polygons overlaid on the density map")
    parser.add_argument('--plot_gt', default=True, type=bool, help="plot ground truth floorplan")

    parser.add_argument('--filename', default='test', type=str)


    return parser

@torch.no_grad()
def inference(model, filename, device, output_dir, plot_pred=True, plot_density=True, semantic_rich=False):
    options = MCSSOptions()
    opts = options.parse()
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gray_img = Image.open(filename).convert("L")

    img = np.array(gray_img)
    #img = cv2.Canny(img, 100, 200)

    grad_X = cv2.Sobel(img, -1, 1, 0)

    grad_Y = cv2.Sobel(img, -1, 0, 1)

    img = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)


    cv2.imwrite("a.jpg", img)
    gray_img.save("d.jpg")

    scene_ids = [filename]
    batched_inputs = [(1/255) * torch.as_tensor(np.ascontiguousarray(np.expand_dims(img, 0)))]
    samples = [x.to(device) for x in batched_inputs]
    
    outputs = model(samples)
    pred_logits = outputs['pred_logits']
    pred_corners = outputs['pred_coords']
    #fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners
    fg_mask = torch.sigmoid(pred_logits) > 0.1 # select valid corners
    if 'pred_room_logits' in outputs:
        prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
        _, pred_room_label = prob[..., :-1].max(-1)
    # process per scene
    for i in range(pred_logits.shape[0]):
        print("Running Evaluation for scene %s" % scene_ids[i])

        fg_mask_per_scene = fg_mask[i]
        pred_corners_per_scene = pred_corners[i]
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
            pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

        # process per room
        for j in range(fg_mask_per_scene.shape[0]):
            fg_mask_per_room = fg_mask_per_scene[j]
            pred_corners_per_room = pred_corners_per_scene[j]
            valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
            if len(valid_corners_per_room)>0:
                corners = (valid_corners_per_room * 255).cpu().numpy()
                corners = np.around(corners).astype(np.int32)

                if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                else:
                    # regular rooms
                    if pred_room_label_per_scene[j] not in [16,17]:
                        if len(corners)>=4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                            room_types.append(pred_room_label_per_scene[j])
                    # window / door
                    elif len(corners)==2:
                        window_doors.append(corners)
                        window_doors_types.append(pred_room_label_per_scene[j])

        if plot_pred:
            if semantic_rich:
                # plot predicted semantic rich floorplan
                pred_sem_rich = []
                for j in range(len(room_polys)):
                    temp_poly = room_polys[j]
                    temp_poly_flip_y = temp_poly.copy()
                    temp_poly_flip_y[:,1] = 255 - temp_poly_flip_y[:,1]
                    pred_sem_rich.append([temp_poly_flip_y, room_types[j]])
                for j in range(len(window_doors)):
                    temp_line = window_doors[j]
                    temp_line_flip_y = temp_line.copy()
                    temp_line_flip_y[:,1] = 255 - temp_line_flip_y[:,1]
                    pred_sem_rich.append([temp_line_flip_y, window_doors_types[j]])

                pred_sem_rich_path = os.path.join(output_dir, '{}_sem_rich_pred.png'.format(scene_ids[i]))
                plot_semantic_rich_floorplan(pred_sem_rich, pred_sem_rich_path, prec=None, rec=None)
            else:
                # plot regular room floorplan
                room_polys = [np.array(r) for r in room_polys]
                floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
                cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), floorplan_map)

        if plot_density:         
            _gray_img = np.array(Image.open(scene_ids[i]).convert("L").resize((256, 256)))
            sample_i = (1/255) * torch.as_tensor(np.ascontiguousarray(np.expand_dims(_gray_img, 0)))
            density_map = np.transpose((sample_i * 255).numpy(), [1, 2, 0])
            #density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
            density_map = np.repeat(density_map, 3, axis=2)
            pred_room_map = np.zeros([256, 256, 3])

            for room_poly in room_polys:
                pred_room_map = plot_room_map(room_poly, pred_room_map)

            # plot predicted polygon overlaid on the density map

            pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args, train=False)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    '''
    # build dataset and dataloader
    dataset_eval = build_dataset(image_set=args.eval_set, args=args)
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval,
                                 drop_last=False, collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                 pin_memory=True)

    for n, p in model.named_parameters():
        print(n)
    '''
    output_dir = Path(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)
    """
    evaluate_floor(
                   model, args.dataset_name, data_loader_eval, 
                   device, save_dir, 
                   plot_pred=args.plot_pred, 
                   plot_density=args.plot_density, 
                   plot_gt=args.plot_gt,
                   semantic_rich=args.semantic_classes>0
                   )
    """
    inference(
            model=model,
            filename=args.filename,
            device=device,
            output_dir=save_dir, 
            plot_pred=args.plot_pred, 
            plot_density=args.plot_density, 
            semantic_rich=args.semantic_classes>0
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RoomFormer evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
