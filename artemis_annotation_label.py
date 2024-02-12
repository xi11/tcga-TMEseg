import os
from glob import glob
import numpy as np
from PIL import Image
import shutil


src_path = '/Volumes/yuan_lab/TIER2/artemis_lei/serial/fine_tune_tme/patch768digital'
dst_mask = '/Volumes/yuan_lab/TIER2/artemis_lei/serial/fine_tune_tme/patch768digital-tcga'
os.makedirs(dst_mask, exist_ok=True)

files = sorted(glob(os.path.join(src_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    mask = np.array(Image.open(file))
    condition = np.isin(mask, [1, 2, 3, 4, 5])
    mask[~condition] = 0
    mask[mask == 1] = 6
    mask[mask == 3] = 1
    mask[mask == 4] = 7
    mask[mask == 2] = 4
    mask[mask == 7] = 2

    #artemis raw label      new label
    #parenchyma 1               6
    #necrosis 2                 4
    #tumor 3                    1
    #stroma 4                   2
    #fat 5                      5

    #tcga-v2: 1,2 ,3, 4,5,6,9,10,11,12,13,14,16, 17, 18, 19, 20
    #tumor: 1, including 1, 19, 20
    #stroma: 2, including 2, 14, 16
    #immune: 3 including 3,10, 11
    #necrosis: 4 including 4, 5, 6, 12
    #fat: 5, including 9
    #normal: 6 including 13, 17
    #blood_vessel: 7, including 18


    mask = Image.fromarray(mask)
    mask.save(os.path.join(dst_mask, file_name))


