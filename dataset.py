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
import numpy as np
import os
import cv2
import math
import random
#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    
    boxes = np.zeros((layers[0]**2+layers[1]**2+layers[2]**2 + layers[3]**2, 4, 8))
    for w in range(layers[0]):
        for h in range(layers[0]):
            boxes = fill_box(boxes,h, w, small_scale, large_scale, 10,0, 0, 0.05)
    for w in range(layers[1]):
        for h in range(layers[1]):
            boxes = fill_box(boxes,h, w, small_scale, large_scale, 5,1, 10*10, 0.1)
    for w in range(layers[2]):
        for h in range(layers[2]):
            boxes = fill_box(boxes,h, w, small_scale, large_scale, 3,2,10*10+5*5, 0.33)
    for w in range(layers[3]):
        for h in range(layers[3]):
            boxes = fill_box(boxes,h, w, small_scale, large_scale, 1,3,10*10+5*5+3*3, 0.5)

    boxes = boxes.reshape(-1,8)
    return boxes
def fill_box(boxes, h, w, ssize,lsize, max_w, layer_num,off_set, center_offset):
    center_x = w/max_w+center_offset
    center_y = h/max_w+center_offset
    #print(w)
    #print(max_w)
    width_1 = ssize[layer_num]
    width_2 = lsize[layer_num]
    width_3 = lsize[layer_num]*np.sqrt(2)
    width_4 = lsize[layer_num]/np.sqrt(2)
    height_1 = ssize[layer_num]
    height_2 = lsize[layer_num]
    height_3 = lsize[layer_num]/np.sqrt(2)
    height_4 = lsize[layer_num]*np.sqrt(2)
    boxes[h*max_w+w+off_set, 0, :] = [center_x, center_y, width_1, height_1, center_x-width_1/2, center_y - height_1/2, center_x+width_1/2, center_y + height_1/2]
    boxes[h*max_w+w+off_set, 1, :] = [center_x, center_y, width_2, height_2, center_x-width_2/2, center_y - height_2/2, center_x+width_2/2, center_y + height_2/2]
    boxes[h*max_w+w+off_set, 2, :] = [center_x, center_y, width_3, height_3, center_x-width_3/2, center_y - height_3/2, center_x+width_3/2, center_y + height_3/2]
    boxes[h*max_w+w+off_set, 3, :] = [center_x, center_y, width_4, height_4, center_x-width_4/2, center_y - height_4/2, center_x+width_4/2, center_y + height_4/2]                                                                                      

    return boxes
#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)

def iou_nms(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    inter = np.maximum(np.minimum(boxs_default[:,2],x_max)-np.maximum(boxs_default[:,0],x_min),0)*np.maximum(np.minimum(boxs_default[:,3],y_max)-np.maximum(boxs_default[:,1],y_min),0)
    area_a = (boxs_default[:,2]-boxs_default[:,0])*(boxs_default[:,3]-boxs_default[:,1])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    #print(x_min,x_max,y_min,y_max)
    x_min_iou =  x_min 
    x_max_iou =  x_min + x_max
    y_min_iou =  y_min 
    y_max_iou =  y_min + y_max
    ious = iou(boxs_default, x_min_iou,y_min_iou,x_max_iou,y_max_iou)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence

    indices = np.where(ious_true == True)
    #print(indices.shape)
    #print(lenindices)
    for i in indices[0]:
        #print(i)
        ann_confidence[i, 3] = 0
        ann_confidence[i, cat_id] = 1
        p_x = boxs_default[i, 0]
        p_y = boxs_default[i, 1]
        g_x = x_min + 0.5*x_max
        g_y = y_min + 0.5*y_max
        p_w = boxs_default[i, 2]
        p_h = boxs_default[i, 3]
        g_w = x_max
        g_h = y_max
        t_x = (g_x - p_x)/p_w
        t_y = (g_y - p_y)/p_h
        t_w = np.log(g_w/p_w)
        t_h = np.log(g_h/p_h)
        ann_box[i,:] = [t_x,t_y,t_w,t_h] 
        #print(ann_box[i,:])
    #update ann_box and ann_confidence (do the same thing as above)
    if len(indices[0]) == 0:
        index_max = np.argmax(ious)

        ann_confidence[index_max, 3] = 0
        ann_confidence[index_max, cat_id] = 1
        p_x = boxs_default[index_max, 0]
        p_y = boxs_default[index_max, 1]
        g_x = x_min + 0.5*x_max
        g_y = y_min + 0.5*y_max
        p_w = boxs_default[index_max, 2]
        p_h = boxs_default[index_max, 3]
        g_w = x_max 
        g_h = y_max
        t_x = (g_x - p_x)/p_w
        t_y = (g_y - p_y)/p_h
        t_w = np.log(g_w/p_w)
        t_h = np.log(g_h/p_h)
        #print(g_h,p)
        ann_box[index_max,:] = [t_x,t_y,t_w,t_h] 

    return ann_box,ann_confidence
    
    



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, test = False, image_size=320, crop = False):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        self.test = test
        self.crop = crop
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.image_size = image_size
        if(train == True):
            self.img_names = os.listdir(self.imgdir)
            #self.img_names = self.img_names[0:int(0.8*len(self.img_names))]
            self.img_names = os.listdir(self.imgdir)
        else:
            self.img_names = os.listdir(self.imgdir)
            #self.img_names = self.img_names[int(0.8*len(self.img_names)):len(self.img_names)]
            self.img_names = os.listdir(self.imgdir)
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> catself.boxs_default,self.threshold,
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        image_name = self.img_names[index]
        #print(ann_name_r)
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = cv2.imread(img_name)
        h,w = image.shape[0],image.shape[1]

        image_ = cv2.resize(image, (self.image_size,self.image_size))
        image_ = image_.transpose(2, 0, 1)
        image  = image.transpose(2, 0, 1)
        if self.train == True:
            if self.crop == False:
                annotations_txt = open(ann_name)
                annotations = annotations_txt.readlines()
                #print(len(annotations))
                annotations_txt.close()
                for i in range(len(annotations)):
                    line = annotations[i].split()
                    cat_id = int(line[0])
                    ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,float(line[1])/w,float(line[2])/h,float(line[3])/w,float(line[4])/h)
            else:
                annotations_txt = open(ann_name)
                annotations = annotations_txt.readlines()
                #print(len(annotations))
                annotations_txt.close()
                #for i in range(len(annotations)):
                i = 0
                line = annotations[i].split()
                cat_id = int(line[0])
                x_start = int(float(line[1]))
                y_start = int(float(line[2]))
                x_end = x_start + int(float(line[3]))
                y_end = y_start + int(float(line[4]))
                x_min = 0
                y_min = 0
                x_max = w
                y_max = h
                img_new_x_start = max(0, random.randrange(x_min, x_start)) ## make sure it is a reasonable image
                img_new_y_start = max(0, random.randrange(y_min, y_start))
                img_new_x_end = min(w, random.randrange(x_end, x_max))
                img_new_y_end = min(h, random.randrange(y_end, y_max))
                #new_img_start_end
                image_new_w = img_new_x_end - img_new_x_start
                image_new_h = img_new_y_end - img_new_y_start

                image_new_x_box_start = max(0,(x_start - img_new_x_start)/image_new_w)
                image_new_y_box_start = max(0,(y_start - img_new_y_start)/image_new_h)
                image_new_x_box_end = min(1,(x_end - img_new_x_start)/image_new_w)
                image_new_y_box_end = min(1,(y_end - img_new_y_start)/image_new_h)
                image_new_box_w = image_new_x_box_end - image_new_x_box_start
                image_new_box_h = image_new_y_box_end - image_new_y_box_start
                image = image[:,img_new_y_start:img_new_y_end, img_new_x_start:img_new_x_end]
                image = image.transpose(1, 2, 0)
                image = cv2.resize(image,(320,320))
                image = image.transpose(2, 0, 1)
                ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,image_new_x_box_start,image_new_y_box_start,image_new_box_w,image_new_box_h)
                image_ = image
                #print(image_name)
        return image_, ann_box, ann_confidence, image_name, h ,w