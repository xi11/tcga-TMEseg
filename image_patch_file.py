import os
import random
import pathlib
import numpy as np
from PIL import Image
import math
from time import sleep
import cv2
from glob import glob


def read_img(data_file, label_file):    # to be checked with if file or path
    data = cv2.imread(data_file)
    labels = cv2.imread(label_file, 0)  # label is in .png format
    return data, labels

def get_label_file(data_file,label_path): #get mask with corresponding name
    data_file_name = os.path.basename(data_file)  #split path and get the last one, which is the file name
    label_file_name = data_file_name[:-4]+'.png'
    label_file = os.path.join(label_path, label_file_name)
    return label_file, data_file_name

def write_to_patch(patch_w, data_files, label_path, save_path):
    curr_data_file = data_files
    file_base_name = os.path.basename(data_files)
    #curr_label_file, file_base_name = get_label_file(curr_data_file,label_path)
    data, labels = read_img(curr_data_file, label_path)
    num_patch = extract_patches_img_label(data, labels, save_path, file_base_name, img_patch_h=patch_w, img_patch_w=patch_w, stride_h=patch_w, stride_w=patch_w, label_patch_h=patch_w, label_patch_w=patch_w)
    #print(num_patch)


def extract_patches_img_label(image, label, save_path, file_base_name, img_patch_h=384, img_patch_w=384, stride_h=384, stride_w=384, label_patch_h=384, label_patch_w=384):
    if image.shape[0] < img_patch_h*0.75 or image.shape[1] < img_patch_w*0.75:
        print(file_base_name) #if any of them are less than 384*0.75, too small, don't need, remove

    elif max(image.shape[0], image.shape[1]) < img_patch_w*1.5:
        patch_img = image #if the max of  width and height are less than 384*1.5, then save the entire image, to avoid much overlapping in the dividing; also pass the long-shape images
        cv2.imwrite(os.path.join(save_path, 'image', str(image.shape[0])+'_'+str(image.shape[1])+'_'+file_base_name + '.png'), patch_img)
        mask = label
        cv2.imwrite(os.path.join(save_path, 'maskPng', 'mask_'+str(image.shape[0])+'_'+str(image.shape[1])+'_' + file_base_name + '.png'), mask)

    else:
        img_h = np.size(image, 0)
        img_w = np.size(image, 1)
        num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h) + 1
        num_patches_img_w = math.ceil((img_w - img_patch_w) / stride_w) + 1
        num_patches_img = num_patches_img_h*num_patches_img_w

        iter_tot = 0
        img_patches = np.zeros((num_patches_img, img_patch_h, img_patch_w, image.shape[2]), dtype=image.dtype)
        label_patches = np.zeros((num_patches_img, label_patch_h, label_patch_w), dtype=image.dtype)
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h) + 1)):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w) + 1)):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = max(0, img_h - img_patch_h)
                    end_h = img_h

                if end_w > img_w:
                    start_w = max(0, img_w - img_patch_w)
                    end_w = img_w

                patch_img = image[start_h:end_h, start_w:end_w, :]
                cv2.imwrite(os.path.join(save_path, 'image', file_base_name+'_'+str(iter_tot) + '.png'), patch_img)
                mask = label[start_h:end_h, start_w:end_w]
                cv2.imwrite(os.path.join(save_path, 'maskPng', 'mask_' + file_base_name + '_' + str(iter_tot) + '.png'), mask)
                iter_tot += 1

        return iter_tot




def run(opts_in):
    save_path = opts_in['save_path']
    data_path = opts_in['data_path']
    label_path = opts_in['label_path']

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(str(save_path) + '/image')
        os.makedirs(str(save_path) + '/maskPng')

    files = sorted(glob(os.path.join(data_path, '*.jpg')))
    for file in files:
        file_name = os.path.basename(file)[:-4]
        train_data = file
        label_path_sub = os.path.join(label_path, file_name+'.png')
        write_to_patch(384, data_files=train_data, label_path=label_path_sub, save_path=save_path)



if __name__ == '__main__':
    opts = {
        'save_path': pathlib.Path('/Volumes/xpan7/project/tcga_tnbc/public_train/background/patch384'),
        'data_path': pathlib.Path('/Volumes/xpan7/project/tcga_tnbc/public_train/background/tile_image'),  #main_file_path -> img_path
        'label_path': pathlib.Path('/Volumes/xpan7/project/tcga_tnbc/public_train/background/tile_mask'),

    }

    run(opts_in=opts)