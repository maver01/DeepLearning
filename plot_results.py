# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
#import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
from matplotlib.collections import LineCollection

from mros_data.utils.logger import get_logger
from mros_data.utils.plotting import plot_data, plot_spectrogram
import torch

import util.misc as utils
import heapq

from models import build_model
#from datasets.face import make_face_transforms

import matplotlib.pyplot as plt
import time

import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

from mros_data.datamodule import SleepEventDataModule
from mros_data.datamodule.transforms import STFTTransform
from librosa.display import specshow


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument("--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument("--masks", action="store_true", help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument("--set_cost_class", default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef", default=0.1, type=float, help="Relative classification weight of the no-object class"
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cpu", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser



@torch.no_grad()

def infer2(model, postprocessors, device, output_path, dm):
    '''
    this function plot the pictures with the predicted bounding boxes. 
    Inputs: (automatic from args)
    Outputs: void, shows the images 
    '''
    data_loader_train = dm.train_dataloader()
    data_loader_val = dm.val_dataloader()
    #model.eval()
    duration = 0
    train_ds = dm.train
    batch = train_ds[1]
    record = batch['record']
    data = batch['signal']
    events = batch['events']
    
    for samples, _targets, image_id, *_ in data_loader_train:
        samples = samples.to(device)
        
        # PRINT EVERYTHING
        print("\nIMAGE_ID: ",image_id) # batch size = 1 --> mros-visit1-aa0021_0100
        # ~ print("\nSAMPLES: ", samples) # tensor
        # ~ print("\nSAMPLES LEN: ", samples.size()) # torch.Size([1,2,513,2401])
        # ~ print("\n_targets", _targets) # tensor
        # ~ print()
        # ~ print("\nTargets", targets) # tensor
        
        predicted_targets = model(samples)
        #predicted_targets = model(data)
        # ~ print(predicted_targets.keys()) # dict_keys(['pred_logits, pred_boxes, aux_outputs])
        # ~ print("PRED BOXES", predicted_targets["pred_boxes"])
       
        
        predicted_targets["pred_logits"] = predicted_targets["pred_logits"].cpu()
        predicted_targets["pred_boxes"] = predicted_targets["pred_boxes"].cpu()
        
        probas = predicted_targets['pred_logits'].softmax(-1)[0, :, :-1] # going to assume that the probas contain the probability for each box.
        indexes_box = heapq.nlargest(10, range(len(probas)), key=probas.__getitem__) # indexes_box contains the indexes of the 10 highest probabilities for the boxes
        # ~ print(indexes_box)
        
        predicted_targets_boxes = predicted_targets["pred_boxes"].numpy()
        v_boxes = []
        for i in indexes_box:
            v_boxes.append(predicted_targets_boxes[0][i])
        
        # ~ print(v_boxes) # it contains the 10 most probable boxes
        x_mean = []
        x_duration = []
        out_ev = np.empty((3))
        for i in v_boxes:
            #x_mean.append(i[0])
            #x_duration.append(i[2])
            out_ev = np.vstack((out_ev,np.array([i[0]-i[2]/2, i[0]+i[2]/2, 0.0]))) 
        out_ev = out_ev[1:,:]    
        # horrible code until the end of the function -> desperate time calls for desperate measures
        h5_sample_index = 0 # save the index number (iterate in the next for loop)
        for h5_sample_name in train_ds:  # for every h5 file in the chosen path 
            #if h5_sample_name["record"] == image_id[0]:
            if h5_sample_name["record"] == record:
                
                train_ds.plot_signals(h5_sample_index, channel_names=['Leg L', "Leg R"])
                plt.title(h5_sample_name["record"])
                plt.show()  
                train_ds.plot_output(out_ev,h5_sample_index,channel_names=['Leg L', "Leg R"])
                plt.show() 
                
            h5_sample_index += 1 # increase index, search in the next one
            
        
        print("\n------------\nCHECKPOINT 1\n-------------")            
        #time.sleep(1000)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    
    # MROS code to create the dataloader dm, that contains the train_ds, that contains the signals, that can be plotted as spectrograms
    params = dict(
        data_dir="mros-data-lm/lm",
        batch_size=args.batch_size,
        n_eval=2,
        n_test=2,
        num_workers=args.num_workers,
        seed=2000,
        events={"lm": "Leg movements"},
        window_duration=600,  # seconds
        cache_data=True,
        default_event_window_duration=[15],
        event_buffer_duration=3,
        factor_overlap=2,
        fs=128,
        matching_overlap=0.5,
        n_jobs=-1,
        n_records=10,
        overfit=args.overfit,
        picks=["legl", "legr"],
        # transform=MultitaperTransform(128, 0.5, 35.0, tw=8.0, normalize=True),
        transform=STFTTransform(
            fs=128, segment_size=int(4.0 * 128), step_size=int(0.125 * 128), nfft=1024, normalize=True
        ),
        scaling="robust",
    )
    dm = SleepEventDataModule(**params)
    dm.setup()
    
        
    # plot everything, data_loader_train is added to contain the data related to the signals in train (with --overfit train and val are the same)
    infer2(model, postprocessors, device, args.output_dir, dm) # created above