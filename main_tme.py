import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import math
from glob import glob

from predict_tme_tcga_hpc import generate_tme
from ss1_stich_stroma import ss1_stich
#from ss1_refine import ss1_refine


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to cws data')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='path to save all output files', default=None)
parser.add_argument('-s', '--save_dir_ss1', dest='save_dir_ss1', help='path to save all ss1 files', default=None)
#parser.add_argument('-sf', '--save_dir_ss1_final', dest='save_dir_ss1_final', help='path to save all final files', default=None)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-c', '--color', dest='color_norm', help='color normalization', action='store_false')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=0, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the input', default=768, type=int)
parser.add_argument('-nC', '--number_class', dest='nClass', help='how many classes to segment', default=6, type=int)
parser.add_argument('-sf', '--scale_factor', dest='scale', help='how many times to scale compared to x20', default=0.0625, type=float)
args = parser.parse_args()

######step0: generate cws tiles from single-cell pipeline

######step1: generate growth pattern for tiles
generate_tme(datapath=args.data_dir, save_dir=args.save_dir, file_pattern=args.file_name_pattern, color_norm=args.color_norm, nfile=args.nfile,
            patch_size=args.patch_size, patch_stride=192, nClass=args.nClass)

#######step2: stich to ss1 level
ss1_stich(cws_folder=args.data_dir, annotated_dir=args.save_dir, output_dir=args.save_dir_ss1,  nfile=args.nfile, file_pattern=args.file_name_pattern, downscale=args.scale)

#######step3: refine ss1 mask
#ss1_refine(cws_folder=args.data_dir, ss1_dir=args.save_dir_ss1, ss1_final_dir=args.save_dir_ss1_final, nfile=args.nfile, file_pattern=args.file_name_pattern)