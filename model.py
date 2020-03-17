import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:sel_bbox = np.expand_dims(sel_bbox, axis=0)
    # sel_conf = np.expand_dims(sel_conf, axis=0)  s, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.reshape(-1,4)
    pred_box = pred_box.reshape(-1,4)
    ann_confidence = ann_confidence.reshape(-1,4)
    ann_box = ann_box.reshape(-1,4)
    #print(ann_confidence)
    #CE_loss = F.binary_cross_entropy()
    indices = ann_confidence[:,3]
    noob_indices = indices == 1
    obj_indices = indices == 0
    #print(noob_indices.sum())
    #print(noob_indices.shape)
    L_cls_obj = F.binary_cross_entropy(pred_confidence[obj_indices],ann_confidence[obj_indices]) 
    L_cls_noob = 3 * F.binary_cross_entropy(pred_confidence[noob_indices],ann_confidence[noob_indices])
    L_box = F.smooth_l1_loss(pred_box[noob_indices], ann_box[noob_indices])
    #print(L_box)
    #print(L_cls)
    L_cls = L_cls_obj + L_cls_noob
    L_yolo = L_cls + L_box
    # asdasdsada
    return L_yolo




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        in_1 = 64
        out_1 = 128
        out_2 = 256
        out_3 = 512
        out_4 = 256
        out_5 = 16

        self.conv1 = nn.Conv2d(3, in_1, kernel_size = 3,stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(in_1)
        self.relu1 = nn.ReLU()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_1, in_1, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(in_1),
            nn.ReLU(),
            nn.Conv2d(in_1, in_1, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(in_1),
            nn.ReLU(),
            nn.Conv2d(in_1, out_1, kernel_size = 3,stride = 2, padding = 1),
            nn.BatchNorm2d(out_1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(out_1, out_1, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_1),
            nn.ReLU(),
            nn.Conv2d(out_1, out_1, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_1),
            nn.ReLU(),
            nn.Conv2d(out_1, out_2, kernel_size = 3,stride = 2, padding = 1),
            nn.BatchNorm2d(out_2),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(out_2, out_2, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_2),
            nn.ReLU(),
            nn.Conv2d(out_2, out_2, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_2),
            nn.ReLU(),
            nn.Conv2d(out_2, out_3, kernel_size = 3,stride = 2, padding = 1),
            nn.BatchNorm2d(out_3),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(out_3, out_3, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_3),
            nn.ReLU(),
            nn.Conv2d(out_3, out_3, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_3),
            nn.ReLU(),
            nn.Conv2d(out_3, out_4, kernel_size = 3,stride = 2, padding = 1),
            nn.BatchNorm2d(out_4),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(out_4, out_4, kernel_size = 1,stride = 1, padding = 0),
            nn.BatchNorm2d(out_4),#10x10
            nn.ReLU(),
            nn.Conv2d(out_4, out_4, kernel_size = 3,stride = 2, padding = 1),
            nn.BatchNorm2d(out_4),#5x5
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(out_4, out_4, kernel_size = 1,stride = 1, padding = 0),
            nn.BatchNorm2d(out_4),#5x5
            nn.ReLU(),
            nn.Conv2d(out_4, out_4, kernel_size = 3,stride = 1, padding = 0),
            nn.BatchNorm2d(out_4),#3x3
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(out_4, out_4, kernel_size = 1,stride = 1, padding = 0),
            nn.BatchNorm2d(out_4),#3x3
            nn.ReLU(),
            nn.Conv2d(out_4, out_4, kernel_size = 3,stride = 1, padding = 0),
            nn.BatchNorm2d(out_4),#1x1
            nn.ReLU(),
        )
        self.c0_b_layer = nn.Conv2d(out_4, out_5, kernel_size = 1,stride = 1, padding = 0)
        self.c0_c_layer = nn.Conv2d(out_4, out_5, kernel_size = 1,stride = 1, padding = 0)
        self.c1_b_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.c1_c_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.c2_b_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.c2_c_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.c3_b_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.c3_c_layer = nn.Conv2d(out_4, out_5, kernel_size = 3,stride = 1, padding = 1)
        self.sm = nn.Softmax(2)
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        batch_size = x.shape[0]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        c1 = self.decoder1(x)
        c2 = self.decoder2(c1)
        c3 = self.decoder3(c2)
        x = self.decoder4(c3)
        bboxes = self.c0_b_layer(x)
        confidence = self.c0_c_layer(x)
        c1_bbox = self.c1_b_layer(c1)
        c1_confidence = self.c1_c_layer(c1)
        c1_bbox = c1_bbox.reshape(batch_size,16,100)
        c1_confidence = c1_confidence.reshape(batch_size,16,100)
        c2_bbox = self.c2_b_layer(c2)
        c2_confidence = self.c2_c_layer(c2)
        c2_bbox = c2_bbox.reshape(batch_size,16,25)
        c2_confidence = c2_confidence.reshape(batch_size,16,25)
        c3_bbox = self.c3_b_layer(c3)
        c3_confidence = self.c3_c_layer(c3)
        c3_bbox = c3_bbox.reshape(batch_size,16,9)
        c3_confidence = c3_confidence.reshape(batch_size,16,9)
        bboxes = bboxes.reshape(batch_size, 16, 1)
        confidence = confidence.reshape(batch_size, 16, 1)
        #print(c3_bbox.shape)
        bboxes = torch.cat((c1_bbox,c2_bbox,c3_bbox,bboxes),2)
        confidence = torch.cat((c1_confidence, c2_confidence,c3_confidence,confidence),2)
        #print(confidence.shape,bboxes.shape)
        bboxes = bboxes.permute(0,2,1)
        confidence = confidence.permute(0,2,1)
        bboxes = bboxes.reshape(batch_size,540,4)
        confidence = confidence.reshape(batch_size,540,4)
        confidence = self.sm(confidence)
        return confidence,bboxes









