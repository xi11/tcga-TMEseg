import os
from glob import glob
import numpy as np
from PIL import Image
import shutil

src_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/mask_digital'
ref_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/v1_part/mask_color_rotate'
dst_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/mask_digital_rotate'
os.makedirs(dst_path, exist_ok=True)

files = sorted(glob(os.path.join(ref_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    src_file = os.path.join(src_path, file_name)
    dst_file = os.path.join(dst_path, file_name)
    shutil.move(src_file, dst_file)

'''
src_path = '/Users/xiaoxipan/Documents/project/public_data/bcss/masks'
dst_color = '/Users/xiaoxipan/Documents/project/public_data/bcss/mask_color'
dst_mask = '/Users/xiaoxipan/Documents/project/public_data/bcss/mask_digital'
os.makedirs(dst_color, exist_ok=True)
os.makedirs(dst_mask, exist_ok=True)

color_mapping = {
    1: (128, 0, 0),
    2: (255, 255, 0),
    3: (255, 0, 0),
    4: (255, 0, 255),
    5: (128, 128, 0),
    6: (0, 255, 255),
    7: (0, 0, 255)
}

files = sorted(glob(os.path.join(src_path, '*.png')))
for file in files:
    file_name = os.path.basename(file)
    mask = np.array(Image.open(file))
    condition = np.isin(mask, [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20])
    mask[~condition] = 0
    mask[(mask == 5) | (mask == 6) | (mask == 12)] = 4
    mask[mask == 9] = 5
    mask[(mask == 10) | (mask == 11)] = 3
    mask[(mask == 13) | (mask == 17)] = 6
    mask[(mask == 14) | (mask == 16)] = 2
    mask[mask == 18] = 7
    mask[(mask == 19) | (mask == 20)] = 1



    
    label                   GT_code
    outside_roi                 0
    tumor                       1
    stroma                      2
    lymphocytic_infiltrate      3
    necrosis_or_debris          4
    glandular_secretions        5
    blood                       6
    exclude                     7
    metaplasia_NOS              8
    fat                         9
    plasma_cells                10
    other_immune_infiltrate     11
    mucoid_material             12
    normal_acinus_or_duct       13
    lymphatics                  14
    undetermined                15
    nerve                       16
    skin_adnexa                 17
    blood_vessel                18
    angioinvasion               19
    dcis                        20
    other                       21
    
    v1: 1, 2, 3, 4, 9, 11
    9 -> 5
    11 -> 3
    #tumor	1
    #stroma	2
    #lymphocytic_infiltrate	3
    #necrosis_or_debris	4
    #fat	5
    
    v2: 1,2 ,3, 4,5,6,9,10,11,12,13,14,16, 17, 18, 19, 20
    tumor: 1, including 1, 19, 20
    stroma: 2, including 2, 14, 16
    immune: 3 including 3,10, 11
    necrosis: 4 including 4, 5, 6, 12
    fat: 5, including 9
    normal: 6 including 13, 17
    blood_vessel: 7, including 18
    
    

    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for key, color in color_mapping.items():
        rgb_image[np.where(mask == key)] = color

    image = Image.fromarray(rgb_image)
    image.save(os.path.join(dst_color, file_name))
    mask = Image.fromarray(mask)
    mask.save(os.path.join(dst_mask, file_name))

'''
