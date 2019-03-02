import os
import os.path as osp
import numpy as np
import shutil

VAL_DIR = 'tiny-imagenet-200/val'
VAL_OUTPUT_DIR = 'tiny-imagenet-200/val_out'
VAL_ANNOTATION = 'tiny-imagenet-200/val/val_annotations.txt'

if not osp.isdir(VAL_OUTPUT_DIR):
    os.mkdir(VAL_OUTPUT_DIR)

annotations = np.loadtxt(VAL_ANNOTATION, dtype=str)
for filename, dirname in annotations[:,:2]:
    filename = osp.join(VAL_DIR, 'images', filename)
    dirname = osp.join(VAL_OUTPUT_DIR, dirname)
    if not osp.isdir(dirname):
        os.mkdir(dirname)
    shutil.copy(filename, dirname)
