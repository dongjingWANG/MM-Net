import glob
import os
import shutil

import numpy as np
from shutil import copy

import pandas as pd


for i in range(8):
    for j in ['train','val']:
        os.makedirs(f'/data/zhujinlin/AffectNet/manually/data/{j}/{i}',exist_ok=True)

aff_path = '/data/zhujinlin/AffectNet/manually/data/'
train_path = os.path.join(aff_path, 'train_set/')
val_path = os.path.join(aff_path, 'val_set/')
data = []
index = 0
for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
    idx = os.path.basename(anno).split('_')[0]
    img_path = os.path.join(train_path, f'images/{idx}.jpg')
    label = int(np.load(anno))
    shutil.copy(img_path, aff_path+'/train/'+str(label))
    index += 1
    if index % 100 == 0:
        print("第{}张图片处理完成".format(index))

index = 0
for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
    idx = os.path.basename(anno).split('_')[0]
    img_path = os.path.join(val_path, f'images/{idx}.jpg')
    label = int(np.load(anno))
    shutil.copy(img_path, aff_path+'/val/'+str(label))
    index += 1
    if index % 100 == 0:
        print("第{}张图片处理完成".format(index))

# sum = 0
# for i in range(8):
# list1 = "/data/zhujinlin/AffectNet/manually/train/"+str(i)+"/"
# img_path2 = "/data/zhujinlin/AffectNet/manually/test/"+str(i)+"/"
# list1 = os.listdir("/data/zhujinlin/AffectNet/manually/data/train_set/images")
# list2 = os.listdir("/data/zhujinlin/AffectNet/manually/test/"+str(i)+"/")
# print("{}: {}个".format(i, len(list1)))
# sum += len(list1)
# print(sum)
