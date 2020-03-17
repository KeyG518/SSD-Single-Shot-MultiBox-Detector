import numpy as np
import cv2
from dataset import iou
from dataset import iou_nms
import math 


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    h,w,_ = image.shape
    # print(h,w)
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1pred_box, pred_confidence, sel_def = non_maximum_suppression(pred_confidence,pred_box,boxs_default)
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                p_x = boxs_default[i, 0]
                p_y = boxs_default[i, 1]
                p_w = boxs_default[i, 2]
                p_h = boxs_default[i, 3]
                d_w = ann_box[i, 2]
                d_h = ann_box[i, 3]
                d_x = ann_box[i, 0]
                d_y = ann_box[i, 1]
                g_x = p_w*d_x + p_x
                g_y = p_h*d_y + p_y
                g_w = p_w * np.exp(d_w)
                g_h = p_h * np.exp(d_h)
                start_point = (int((g_x - 0.5*g_w)*w), int((g_y - 0.5*g_h)*h)) #top left corner, x1<x2, y1<y2
                end_point = (int((g_x + 0.5*g_w)*w), int((g_y + 0.5*g_h)*h))  #bottom right corner
                start_point_def = (int((boxs_default[i, 4])*w), int((boxs_default[i, 5])*h))
                end_point_def = (int((boxs_default[i, 6])*w), int((boxs_default[i, 7])*h))
                # start_point_def = (int((boxs_default[i, 4])*w), int((boxs_default[i, 5])*w))
                # end_point_def = (int((boxs_default[i, 6])*w), int((boxs_default[i, 7])*w))
                
                color = colors[j] #use red green blue to rannbox
                thickness = 2
                cv2.rectangle(image1, start_point, end_point, color, thickness)
                cv2.rectangle(image2, start_point_def, end_point_def, color, thickness)
                #   print("hello",start_point,end_point)
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.7:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                #print(pred_box.shape)
                p_x = boxs_default[i, 0]
                p_y = boxs_default[i, 1]
                p_w = boxs_default[i, 2]
                p_h = boxs_default[i, 3]
                d_w = ann_box[i, 2]
                d_h = ann_box[i, 3]
                d_x = ann_box[i, 0]
                d_y = ann_box[i, 1]
                g_x = p_w*d_x + p_x
                g_y = p_h*d_y + p_y
                g_w = p_w * np.exp(d_w)
                g_h = p_h * np.exp(d_h)
                start_point = (int((g_x - 0.5*g_w)*w), int((g_y - 0.5*g_h)*h)) #top left corner, x1<x2, y1<y2
                end_point = (int((g_x + 0.5*g_w)*w), int((g_y + 0.5*g_h)*h))  #bottom right corner
                start_point_def = (int((boxs_default[i, 4])*w), int((boxs_default[i, 5])*h))
                end_point_def = (int((boxs_default[i, 6])*w), int((boxs_default[i, 7])*h))
                color = colors[j] #use red green blue to rannbox
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)
                cv2.rectangle(image4, start_point_def, end_point_def, color, thickness)
    
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(10000)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
def visualize_pred_nms(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default,o_h,o_w):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    h,w,_ = image.shape
    # print(h,w)
    list_box_start = []
    list_box_end = []
    list_class = []
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1pred_box, pred_confidence, sel_def = non_maximum_suppression(pred_confidence,pred_box,boxs_default)
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                p_x = boxs_default[i, 0]
                p_y = boxs_default[i, 1]
                p_w = boxs_default[i, 2]
                p_h = boxs_default[i, 3]
                d_w = ann_box[i, 2]
                d_h = ann_box[i, 3]
                d_x = ann_box[i, 0]
                d_y = ann_box[i, 1]
                g_x = p_w*d_x + p_x
                g_y = p_h*d_y + p_y
                g_w = p_w * np.exp(d_w)
                g_h = p_h * np.exp(d_h)
                start_point = (int((g_x - 0.5*g_w)*w), int((g_y - 0.5*g_h)*h)) #top left corner, x1<x2, y1<y2
                end_point = (int((g_x + 0.5*g_w)*w), int((g_y + 0.5*g_h)*h))  #bottom right corner
                start_point_def = (int((boxs_default[i, 4])*w), int((boxs_default[i, 5])*h))
                end_point_def = (int((boxs_default[i, 6])*w), int((boxs_default[i, 7])*h))
                # start_point_def = (int((boxs_default[i, 4])*w), int((boxs_default[i, 5])*w))
                # end_point_def = (int((boxs_default[i, 6])*w), int((boxs_default[i, 7])*w))
                
                color = colors[j] #use red green blue to rannbox
                thickness = 2
                cv2.rectangle(image1, start_point, end_point, color, thickness)
                cv2.rectangle(image2, start_point_def, end_point_def, color, thickness)
    pred_box, pred_confidence, sel_def = non_maximum_suppression(pred_confidence,pred_box,boxs_default)
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                #print(pred_box.shape)
                start_point = (int(pred_box[i,0]), int(pred_box[i,1]))
                end_point = (int(pred_box[i,2]), int(pred_box[i,3]))
                f_x_s = pred_box[i,0]/320
                f_y_s = pred_box[i,1]/320
                f_x_e = pred_box[i,2]/320
                f_y_e = pred_box[i,3]/320
                f_x_s_o = f_x_s*o_w
                f_y_s_o = f_y_s*o_h
                f_x_width_o = f_x_e*o_w - f_x_s_o
                f_y_height_o = f_y_e*o_h -f_y_s_o
                f_start = (f_x_s_o,f_y_s_o)
                f_end = (f_x_width_o, f_y_height_o)
                #f_start = f_start.detach().cpu().numpy()
                #f_end = f_end.detach().cpu().numpy()
                if (f_start not in list_box_start) and (f_end not in list_box_end):
                    list_box_start.append(f_start)
                    list_box_end.append(f_end)
                    list_class.append(j)
                start_point_def = (int((sel_def[i, 4])*w), int((sel_def[i, 5])*w))
                end_point_def = (int((sel_def[i, 6])*w), int((sel_def[i, 7])*w))
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image3, start_point, end_point, color, thickness)
                cv2.rectangle(image4, start_point_def, end_point_def, color, thickness)
                
    
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    #cv2.imshow("test"+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    #save_dir_img = "data/test/output/out_image_train/"+ windowname
    save_dir_anno = "data/test/output/out_test_anno/"+ windowname[:-3] + "txt"
    #print("hello")
    #cv2.imwrite(save_dir_img, image)
    f = open(save_dir_anno,"w+")
    for i in range (len(list_box_start)):
        f.write(str(list_class[i]) + " " + str(list_box_start[i][0]) + " " + str(list_box_start[i][1]) + " " + str(list_box_end[i][0]) + " " + str(list_box_end[i][1])+ "\n")
    #cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.15, threshold=0.65):
#     #input:
#     #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
#     #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4] A 
#     #boxs_default -- default bounding boxes, [num_of_boxes, 8]
#     #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
#     #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
#     #output:
#     #depends on your implementation.
#     #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
#     #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    sel_bbox = []
    sel_conf = [] 
    sel_def = []

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    p_x = boxs_default[:, 0]
    p_y = boxs_default[:, 1]
    p_w = boxs_default[:, 2]
    p_h = boxs_default[:, 3]
    d_x = box_[:, 0] 
    d_y = box_[:, 1]
    d_w = box_[:, 2]
    d_h = box_[:, 3]
    g_x = p_w*d_x + p_x
    g_y = p_h*d_y + p_y
    g_w = p_w * np.exp(d_w)
    g_h = p_h * np.exp(d_h)
    x_min = (g_x - g_w*0.5)*320
    y_min = (g_y - g_h*0.5)*320
    x_max = (g_x + g_w*0.5)*320
    y_max = (g_y + g_h*0.5)*320
    box_[:,0] = x_min
    box_[:,1] = y_min
    box_[:,2] = x_max
    box_[:,3] = y_max
    score = confidence_[:,0:3]
    num_classes = score.shape[1]
    #print(num_classes)
    

    for class_idx in range(0, num_classes):
        # choose candidate
        scores = score[:, class_idx]
        sorted_score = np.flip(np.sort(scores))
        indexes = np.flip(np.argsort(scores))
        indexes = indexes[sorted_score > threshold]
        picked = []
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            indexes = indexes[1:]
            rest_boxes = box_[indexes]
            #print(x_min[current])
            ious = iou_nms(rest_boxes, x_min[current],y_min[current],x_max[current],y_max[current])
            indexes = indexes[ious < overlap]

        for i in range(len(picked)):
            sel_bbox.append(box_[picked[i]])
            sel_conf.append(confidence_[picked[i]])
            sel_def.append(boxs_default[picked[i]])
    sel_bbox = np.asarray(sel_bbox)
    sel_conf = np.asarray(sel_conf)
    sel_def = np.asarray(sel_def)      
    return sel_bbox,sel_conf,sel_def










