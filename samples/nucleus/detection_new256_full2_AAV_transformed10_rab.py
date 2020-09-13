import nucleus_ratRabies as  nucleus1
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import fnmatch
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import tensorflow as tf
import keras.backend as K
# from imgaug import augmenters as iaa
from PIL import Image
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os
from skimage.io import imread
from multiprocessing import Pool
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
from skimage.morphology import convex_hull_image
from functools import partial
from time import gmtime, strftime
import matplotlib.image as mpimg
import pickle
import matplotlib.path as mplPath
import sys
from skimage.io import imread
from scipy import misc 
from scipy.io import loadmat
import h5py
import hdf5storage
import bisect
import statistics
from scipy.ndimage.morphology import binary_fill_holes




def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    #STEPS_PER_EPOCH = (35 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    #VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.9

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 4000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0, 0, 0])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_CONFIDENCE = 0.9



def image2mask(image, maskB, opB,  w, h):
    window = 512
    # count = 0
    for row in range(0, w - window, window):
        for col in range(0, h - window, window):
            # print(row, col)
            # print(row+window, col+window)
            # maskR = np.zeros((window, window), dtype='bool')
            # maskG = np.zeros((window, window), dtype='bool')
            # mask = np.zeros((window, window), dtype='uint8')
            tile = image[row:row + window, col:col + window, :]
            tileM = maskB[row:row + window, col:col + window] 
            if np.sum(tileM): 
                out = nucleus1.detect(model, tile)
                # print(out.shape)
                outK = np.zeros((window, window), dtype= np.uint8)
                outK[np.sum(out,axis=2).nonzero()] = 1
                outK = cv2.threshold(outK, 0, 1, cv2.THRESH_BINARY)
                outK = np.asarray(outK[1])
                outK = binary_fill_holes(outK)
                opB[row:row + window, col:col + window] = outK
        
    return opB

weights_path = '/home/samik/Mask_RCNN/logs/nucleus20200304T1620/mask_rcnn_nucleus_0510.h5'

os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

model.load_weights(weights_path, by_name=True)

brainNo = 'PTM800'

filePath = '/nfs/data/main/M25/mba_converted_imaging_data/PTM800/PTM800_rename/'
# filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/' #reg_high_tif_pad_jp2/'
# maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'
maskDir = '/nfs/data/main/M32/Samik/PTM800/mask/'

outDirR = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_unreg/' + brainNo + '/'
jsonOutB = os.path.join(outDirR, 'json/')
maskOut = os.path.join(outDirR, 'mask/')


os.system("mkdir " + outDirR)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutB)
# os.system("mkdir " + jsonOutG)

fileList1 = os.listdir(filePath)
fileList2 = [] #os.listdir(maskOut)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*IHC*')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
        if fichier in fileList2[:]:
            fileList1.remove(fichier)

#print(fileList1)

for files in fileList1:
    print(files)
    image = imread_fast(os.path.join(filePath, files))
    w, h, c = image.shape
    maskB = cv2.imread(os.path.join(maskDir,files.replace('jp2', 'png')), cv2.IMREAD_UNCHANGED)
    maskB = cv2.resize(maskB, (h,w))
    maskB = maskB / maskB.max()
    print(image.shape, maskB.shape)

    ret,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
    maskB = np.uint8(maskB) * 255        

    opB = np.zeros((w,h),dtype='bool')
    opB = image2mask(image, maskB, opB, w, h)
    opB = np.uint8(opB) * 255


    _, thresh = cv2.threshold(opB,127,255,cv2.THRESH_BINARY)
    _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
    f = open(os.path.join(jsonOutB, files.replace('jp2', 'json')), "w")
    f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"0\",\"properties\":{\"name\":\"Red Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":[")
    for pts in centroids:
        pts64 = pts.astype(np.int64)
        f.write("[" + str(pts64[0]) + "," + str(pts64[1]*-1) + "],")
    f.write("[]]}}]}")
    f.close()

    cv2.imwrite(os.path.join(maskOut, files), opB)
    # cv2.imwrite(os.path.join(maskOut, 'R_' + files), opR)
    # cv2.imwrite(os.path.join(maskOut, 'G_' + files), opG)







