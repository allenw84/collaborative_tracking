import os
import sys
import json
import glob
from pathlib import Path

import cv2
import numpy as np
import time

#DATA_DIR = Path('../kaggle/input')
ROOT_DIR = Path('kaggle/working')
FASION_DIR = Path('Mask_RCNN/fasion/working')

NUM_CATS = 46
IMAGE_SIZE = 512

#os.chdir('Mask_RCNN')
#sys.path.append(str(ROOT_DIR) + '/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
from keras import backend as K
# print(mrcnn.__file__)

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# def model_setup(self):
#     FASION_DIR = Path('Mask_RCNN/fasion/working')
#     NUM_CATS = 46
#     IMAGE_SIZE = 512

    
#     with open(str(FASION_DIR) + "/label_descriptions.json") as f:
#         label_descriptions = json.load(f)
#     label_names = [x['name'] for x in label_descriptions['categories']]

#     glob_list = glob.glob('Mask_RCNN/fasion/working/fashion*/mask_rcnn_fashion_0007.h5') #four folders

#     model_path = glob_list[numw] if glob_list else ''

#         class InferenceConfig(FashionConfig):
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1

#         inference_config = InferenceConfig()BATCH_SIZE

#         model = modellib.MaskRCNN(mode='inference', 
#                                 config=inference_config,
#                                 model_dir=str(ROOT_DIR))

#         assert model_path != '', "Provide path to trained weights"
#         print("Loading weights from ", model_path)
#         model.load_weights(model_path, by_name=True)
    
#     return model

def mask_ouput(img):

    FASION_DIR = Path('Mask_RCNN/fasion/working')
    NUM_CATS = 46
    IMAGE_SIZE = 512
    

    
    with open(str(FASION_DIR) + "/label_descriptions.json") as f:
        label_descriptions = json.load(f)
    label_names = [x['name'] for x in label_descriptions['categories']]
    

    def resize_image(img):
        #img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
        return img

    glob_list = glob.glob('Mask_RCNN/fasion/working/fashion*/mask_rcnn_fashion_0007.h5') #four folders

    detected_result = {}
    for numw in range(0, 4):
        model_path = glob_list[numw] if glob_list else ''

        class InferenceConfig(FashionConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        

        inference_config = InferenceConfig()

        model = modellib.MaskRCNN(mode='inference', 
                                config=inference_config,
                                model_dir=str(ROOT_DIR))

        assert model_path != '', "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        # Since the submission system does not permit overlapped masks, we have to fix them
        def refine_masks(masks, rois):
            areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
            mask_index = np.argsort(areas)
            union_mask = np.zeros(masks.shape[:-1], dtype=bool)
            for m in mask_index:
                masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
                union_mask = np.logical_or(masks[:, :, m], union_mask)
            for m in range(masks.shape[-1]):
                mask_pos = np.where(masks[:, :, m]==True)
                if np.any(mask_pos):
                    y1, x1 = np.min(mask_pos, axis=1)
                    y2, x2 = np.max(mask_pos, axis=1)
                    rois[m, :] = [y1, x1, y2, x2]
            return masks, rois

        #image_path = 'kaggle/input/4k-camera/phone/record/testimg1.PNG' #input 

        #img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = model.detect([resize_image(img)])
        r = result[0]
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            y_scale = img.shape[0]/IMAGE_SIZE
            x_scale = img.shape[1]/IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']

        for i in range(r['class_ids'].shape[0]):
            detected_result[label_names[r['class_ids'][i]-1]] = [r['scores'][i], rois]

        ans_arr = []
    
        for i in detected_result:
            ans_arr.append([i, detected_result[i][0], detected_result[i][1][0].tolist()])
        
        K.clear_session()

    return ans_arr
# {'top, t-shirt, sweatshirt': [0.8528358, array([[ 48,  24, 122,  81]])], 
#  'shoe': [0.938905, array([[190,  38, 202,  51]])]}

if __name__ == '__main__':
    image_path = 'kaggle/input/4k-camera/phone/record/testimg1.PNG' #input 
    img = cv2.imread(image_path)
    print(mask_ouput(img))