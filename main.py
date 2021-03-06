import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 16


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
#print(boxs_default[0,:])
#print(boxs_default[1,:])
#print(boxs_default[2,:])
#print(boxs_default[3,:])
#print(boxs_default[4,:])



#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True
crop_mode = False

#if False:
if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    #dataset = COCO("data/test/output/out_image/", "data/test/output/out_anno/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataset_crop = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320, crop = True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_crop = torch.utils.data.DataLoader(dataset_crop, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAIN
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, _, _, _= data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
            #print(avg_loss/avg_count)
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        if crop_mode :
            for i, data in enumerate(dataloader_crop, 0):
                images_, ann_box_, ann_confidence_, _, _,_= data
                images = images_.cuda()
                ann_box = ann_box_.cuda()
                ann_confidence = ann_confidence_.cuda()

                optimizer.zero_grad()
                pred_confidence, pred_box = network(images)
                loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
                loss_net.backward()
                optimizer.step()
                
                avg_loss += loss_net.data
                avg_count += 1
                #print(avg_loss/avg_count)
                pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
                pred_box_ = pred_box[0].detach().cpu().numpy()
                visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
                print('[%d] time: %f train loss with crop: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))

        
        #visualize
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #TEST
        network.eval()
        
        # TODO: split the dataset into 80% training and 20% testing
        # use the training set to train and the testing set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, _ ,_, _= data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth%d' %epoch)

        
#         #TODO: save predicted bounding boxes and classes to a txt file.
#         #you will need to submit those files for grading this assignment
#         # print(os.getcwd())
#         # annotations_txt = open(annname, 'w')
#         # for i in range(len(pred_confidence)):
#         #     for j in range(class_num):
#         #         if pred_confidence[i,j]>0.5:
#         #             start_point_x = int(pred_box[i,0])
#         #             start_point_y = int(pred_box[i,1])
#         #             end_point_x = int(pred_box[i,2]) - int(pred_box[i,0])
#         #             end_point_y = int(pred_box[i,3]) - int(pred_box[i,1])
#         #             annotations_txt.write("%d %d %d %d " % j,start_point_x,  start_point_y,  end_point_x, end_point_y)
#         #             annotations_txt.close
        
else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth49'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, s_name,h,w= data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        h = h.detach().cpu().numpy()[0]
        w = w.detach().cpu().numpy()[0]
        
        #pred_confidence_,pred_box_,boxs_default_back = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        visualize_pred_nms(s_name[0], pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default, h, w)
        cv2.waitKey(1000)



