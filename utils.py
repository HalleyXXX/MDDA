# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:00:36 2022

@author: GUI
"""

import os
import math
import random
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import params
from datasets import get_H_data, get_L_data, get_data_test

def get_data_loader(name, test_flag=False, kind='train', train_num=None, sample_num=None):
    """ Get data"""
    if name == 'bone_H_'+str(params.input_h):
        if test_flag:
            return get_data_test(params.src_data_path, kind)
        else:
            return get_H_data(kind)
    elif name == 'bone_L_'+str(params.input_h):
        if test_flag:
            return get_data_test(params.tgt_data_path, kind)
        else:
            return get_L_data(params.tgt_data_path,train_num,kind,sample_num)
    elif name == 'bone_T_256':
        if test_flag:
            return get_data_test(r'inputs\bone_T_256', kind)
        else:
            return get_H_data(kind)
    elif name == 'bone_T_256_L':
        if test_flag:
            return get_data_test(r'inputs/bone_T_256_L', kind)
        else:
            return get_H_data(kind)
    elif name == 'bone_DR_'+str(params.input_h):
        if test_flag:
            return get_data_test(params.DR_data_path, kind)
        else:
            return get_L_data(params.DR_data_path,train_num,kind,sample_num)

def init_model(net, restore=None):
    """Init models with cuda ."""
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore), strict=True)
        net.restored = True
        print(">>Restore model from<<\n {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        with torch.no_grad():
            variable = Variable(tensor)
    else:
        variable = Variable(tensor)
    return variable

def set_requires_grad(model, requires_grad=True, layers=None):
    if layers == None:
        for param in model.parameters():
            param.requires_grad = requires_grad
    else:
        for layer in layers:
            for name, value in layer.named_parameters():
                value.requires_grad = requires_grad
                        
class ImagePool:
    """An image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.

    Args:
        pool_size (int): the size of image buffer, if pool_size=0, no buffer will be created

    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Args:
            images (torch.Tensor): the latest generated images from the generator

        Returns:
            By 50/100, the buffer will return input images.
            By 50/100, the buffer will return images previously stored in the buffer,
            and insert the current images to the buffer.

        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def entropy(imgs):
    numbers = []
    for l in range(imgs.shape[0]):
        img = imgs[l,0,:,:]
        cv2.imwrite(str(l)+'.png', img)
#        print(img.shape)
        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k =  float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if(tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
#        print(res)
        if res < 1.5 and res > 0.8:
            numbers.append(l)
    return numbers

def get_edge(tensor, img_size=(1024, 512), meansub_or_norm='meansub'):
    
    batch_size = tensor[0].size()[0]

    edge_maps = []
    
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
         
    tensor = np.array(edge_maps)
    
    image_shape = [img_size] * batch_size
    
    edge_imgs = torch.zeros([batch_size, 1, img_size[1], img_size[0]])
    
    idx = 0
    
    for i_shape in image_shape:
        
        tmp = tensor[:, idx, ...]
        
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img[tmp_img < 0.0] = 0.0
            tmp_img = 255.0 * (1.0 - tmp_img)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))

            preds.append(tmp_img)
            if i == 6:
                fuse = tmp_img

        fuse = fuse.astype(np.uint8)

        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        
        if(meansub_or_norm == 'meansub'):
            average = average - 229.84
        
        elif(meansub_or_norm == 'norm'):
            average = average / 255
        
        
        edge_imgs[idx] = torch.tensor(average)

        idx += 1

    return edge_imgs
 

if __name__ == '__main__':
    path_1 = r'C:\Users\ROG\Desktop\domain_adaptation\outputs\a_test\target\1'
    files_1 = os.listdir(path_1)
    a = []
    for file in files_1:
        img = cv2.imread(os.path.join(path_1, file), 0)
        img = np.reshape(img, [1,1,256,256])
        entropy_ = entropy(img)
        a.append(entropy_)




