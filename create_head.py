import torch
from torchvision import transforms as ts
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import json
from . loaders import pil_loader,make_dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',".json"
]
HEAD_SIZE=64
HALF_HEAD=64//2
BODY_SIZE=512
transform_dict = {
        'body': ts.Compose(
        [ts.Resize((BODY_SIZE,BODY_SIZE)),
         np.array
         ])}

def save_head(fname,im_fname,im_sample,target_root):
    with open(fname) as f:
        data = json.load(f)
    head_img=np.zeros((HEAD_SIZE,HEAD_SIZE,3))
    head_cent=np.array([0,0])
    for jj in range(len(data["people"])):
        face=np.array(data["people"][jj]["face_keypoints_2d"])# *70 points
        face[face>511]=511
        head_cent=np.array([face[3*30],face[3*30+1]],dtype=np.int)
        head_img=im_sample[head_cent[1]-HALF_HEAD:\
              head_cent[1]+HALF_HEAD,\
              head_cent[0]-HALF_HEAD:\
              head_cent[0]+HALF_HEAD,:]

    head_pil = Image.fromarray(head_img)
    head_pil.save(f"{target_root}/{im_fname[:-4]}_head.png")
    return head_img,head_cent


class create(Dataset):
    def __init__(self,
                img_root: List[str],
                lbl_root: List[str],
                target_root: str,
                ) -> None:
        self.img_root,self.lbl_root,self.target_root=img_root,lbl_root,target_root
        self.body_img,self.body_json,self.imname,_=make_dataset(self.img_root,self.lbl_root)
        self.loader=pil_loader
        self.transform=transform_dict["body"]
        os.makedirs(self.target_root, exist_ok = True)
        
    def __getitem__(self,index:int) ->Any:
        im_path= self.body_img[index]
        tgt_path= self.body_json[index]
        im_sample = self.transform(self.loader(im_path))
        head_img,head_center = save_head(tgt_path,self.imname[index],im_sample,self.target_root)
        return head_img,head_center
    def __len__(self) ->int:
        return len(self.body_img)

