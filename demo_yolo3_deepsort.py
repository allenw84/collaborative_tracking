import os
import tensorflow
import cv2
import time
import argparse
import numpy as np
import math

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

from mtp import subroi, get_bbox_area
import multiprocessing as mp
import multiprocessing #import Process
import path_util as pu
import time
from maskout import *
from checking import *
from color_dete import color, merge_color

import mrcnn.model as modellib
#import maskout.FashionConfig as faconfig
#import maskout.InferenceConfig as inconfig

import pickle
from socket import *

people_path = []
direction_start = []
unseen_frame = []
distance_threshold = 7.5
area_dic={}
results=[]
CAMERA_ID = 'A' #edit by camera

#global user_entry_dict 
user_entry_dict = {} #most important
#example: dict = { 1: [[ [shirt,0.9,blue],[shoe,0.75,red] ], exit_point, 'A', grid_box]}
exix_point = 0 #default value
area_threshold = 100000  

class Detector(object):
    def __init__(self, args):
        self.args = args

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)
        self.class_names = self.yolo3.class_names


        #self.maskrcnn = 

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 30, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def return_user_dict(self):
        return self.user_entry_dict

    def detect(self):
        #xmin, ymin, xmax, ymax = self.area
        jump_flag = 1 
        start = time.time()
        while self.vdo.grab(): 
            #multicore
            #pool = mp.Pool(processes=6) #6-core
            _, ori_im = self.vdo.retrieve()
            im_height, im_width = ori_im.shape[:2]
            x_max = 5
            y_max = 5
            x_grid = int(im_width / x_max)
            y_grid = int(im_height / y_max)
            display_im = ori_im
            
            # for i in range(1, x_max + 1):
            #     cv2.line(ori_im, (x_grid * i, 0), (x_grid * i, im_height), (0, 255, 255), 3)
            # for i in range(1, y_max + 1):
            #     cv2.line(ori_im, (0, y_grid * i), (im_width, y_grid * i), (0, 255, 255), 3)
            # for i in range(len(unseen_frame)):
            #     if unseen_frame[i] > -1:
            #         unseen_frame[i] += 1 
            if jump_flag%2 ==0 : #jump frame  
                #start = time.time()

                clientsocket = socket(AF_INET,SOCK_STREAM)
                clientsocket.connect(('140.114.79.179',10523)) 
                clientsocket.send(pickle.dumps(user_entry_dict))

                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                #img = ori_im
                
                bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
                cv2.circle(ori_im, (3900, 2100), 50, (255,0,0),-1)
                
                if bbox_xcycwh is not None:
                    # select class person
                    mask = cls_ids==0

                    bbox_xcycwh = bbox_xcycwh[mask]
                    bbox_xcycwh[:,3:] *= 1.2

                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                    for output in outputs:
                        if output[4] > len(people_path):
                            for i in range(0, output[4] - len(people_path)):
                                people_path.append([])
                                direction_start.append(0)
                                unseen_frame.append(-1)
                        people_path[output[4] - 1].append(np.array(([(output[0] + output[2]) / 2, output[3]])))
                        coordinate = output[:4]
                        bbox_area = get_bbox_area(coordinate)
                        
                        features = []
                        if bbox_area > area_threshold : 
                            try :
                                if area_dic[output[-1]] < bbox_area :
                                    area_dic[output[-1]] = bbox_area
                                    roiImg = im[output[:4][1]:output[:4][3],output[:4][0]:output[:4][2]] #img[y, x]
                                    features = mask_ouput(roiImg) #features=[[t-shirt, 0.9, [coordination]],...]
                                    features = merge_color(roiImg, features)
                                    #result = pool.apply_async(subroi,(ori_im,output))
                                    #results.append(result)
                                    #for wait()
                                    print("re: ---------------",features)

                            except KeyError:
                                area_dic.setdefault(output[-1],bbox_area)
                                roiImg = im[output[:4][1]:output[:4][3],output[:4][0]:output[:4][2]] #img[y, x]
                                features = mask_ouput(roiImg) #features=[[t-shirt, 0.9, [coordination]],...]
                                features = merge_color(roiImg, features)
                                #result = pool.apply_async(subroi,(ori_im,output))
                                #results.append(result)
                                print("wait---------------")
                            
                            
                            if output[-1] not in user_entry_dict:
                                user_entry_dict.setdefault(output[-1],[features,exix_point,CAMERA_ID,[]]) #add entry id 
                            else:
                                for feature in features:
                                    flag = 1 
                                    for i in range(len(user_entry_dict[output[-1]][0])):
                                        if feature[0] in user_entry_dict[output[-1]][0][i]:
                                            user_entry_dict[output[-1]][0][i][1] = max(user_entry_dict[output[-1]][0][i][1],feature[1]) #update the confodence
                                        flag = 0 
                                    if flag == 1 :
                                        user_entry_dict[output[-1]][0].append(feature)
                            print(user_entry_dict)
                            
                        #call project.py
                            find_grids( output, [x_grid, y_grid], 0.3, user_entry_dict[output[-1]])


                        x = []
                        y = []
                        for i in range(direction_start[output[4] - 1], len(people_path[output[4] - 1])):
                            x.append(people_path[output[4] - 1][i][0])
                            y.append(people_path[output[4] - 1][i][1])
                        path_x = (output[0] + output[2]) / 2
                        path_y = output[3]
                        if(len(x) > 1):
                            a, b, c = pu.cal_simple_linear_regression_coefficients(x, y)
                            #print(abs(a * path_x + b * path_y + c) / math.sqrt(a * a + b * b))
                            if abs(a * path_x + b * path_y + c) / math.sqrt(a * a + b * b) > 200 and unseen_frame[output[4] - 1] < 10:
                                continue;
                            if abs(a * path_x + b * path_y + c) / math.sqrt(a * a + b * b) < distance_threshold:
                                #print("projection")
                                path_x, path_y = pu.find_projection(a, b, c, path_x, path_y)
                                if len(people_path[output[4] - 1]) > 0:
                                    prev_x = people_path[output[4] - 1][len(people_path[output[4] - 1]) - 1][0]
                                    prev_y = people_path[output[4] - 1][len(people_path[output[4] - 1]) - 1][1]
                                    velocity = math.sqrt((path_x - prev_x) * (path_x - prev_x) + (path_y - prev_y) * (path_y - prev_y)) * 30 / (unseen_frame[output[4] - 1] + 1)
                                    #print("velocity: {}".format(velocity))
                            else:
                                #print("turn")
                                direction_start[output[4] - 1] = len(people_path[output[4] - 1])
                        people_path[output[4] - 1].append(np.array((path_x, path_y)))
                        unseen_frame[output[4] - 1] = 0
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                        for id in identities:
                            for i in range(1, len(people_path[id-1])):
                                cv2.line(ori_im, (int(people_path[id-1][i-1][0]), int(people_path[id-1][i-1][1])), 
                                (int(people_path[id-1][i][0]), int(people_path[id-1][i][1])), (0, 0, 255), 3)
                        #pool.close()
                        #pool.join()
                    # for result in results:
                    #     print(result.get())
                #end = time.time()
                #print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
                print(area_dic)
            jump_flag+=1
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(ori_im)
        end = time.time()
        print(end-start)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
