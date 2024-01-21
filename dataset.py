import os

import cv2
import numpy as np
import torch
import torch.utils.data
import params

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, mask_flag=True, max_iters=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.flag = mask_flag
        self.test = False
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.test = True

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
#        print(os.path.join(self.img_dir, img_id + self.img_ext))
        if self.flag:
            mask = []
            try:
                for i in range(self.num_classes):
                    mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
                #数组沿深度方向进行拼接。
                mask = np.dstack(mask)
            except:
                print(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext))
                import sys
                sys.exit()

        if self.transform is not None:
            if self.flag:
                augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
                img = augmented['image']#参考https://github.com/albumentations-team/albumentations
                mask = augmented['mask']
            else:
                augmented = self.transform(image=img)
                img = augmented['image']
    
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        if self.flag:
            mask = mask.astype('float32') / 255
            mask = mask.transpose(2, 0, 1)
            return img, mask, {'img_id': img_id}
        else:
            return img, {'img_id': img_id}
