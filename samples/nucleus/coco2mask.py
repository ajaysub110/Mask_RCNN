import numpy
import cv2
import json
import matplotlib.pyplot as plt
import os
import shutil

from PIL import Image

IMAGE_ROOT = "/home/ajay/ccbr/dataset/data/"
MASK_ROOT = "/home/ajay/ccbr/dataset/masked/"

def decodeSeg(mask, segmentations):
    """
    Draw segmentation
    """
    pts = [
        numpy
            .array(anno)
            .reshape(-1, 2)
            .round()
            .astype(int)
        for anno in segmentations
    ]
    mask = cv2.fillPoly(mask, pts, 1)

    return mask

def decodeRl(mask, rle):
    """
    Run-length encoded object decode
    """
    mask = mask.reshape(-1, order='F')

    last = 0
    val = True
    for count in rle['counts']:
        val = not val
        mask[last:(last+count)] |= val
        last += count

    mask = mask.reshape(rle['size'], order='F')
    return mask

def annotation2binarymask(annotation, w, h):
    mask = numpy.zeros((w, h), numpy.uint8)
    segmentations = annotation['segmentation']
    if isinstance(segmentations, list): # segmentation
        mask = decodeSeg(mask, segmentations)
    else:                               # run-length
        mask = decodeRl(mask, segmentations)
    return mask

def load_images(jsonf):
    f = json.load(open(jsonf))

    return f['images'], f['annotations']

def save_image(im):
    file_name = im['file_name']

    # create image dir if doesn't exist
    image_dir = os.path.join(MASK_ROOT, file_name[:-4], 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # save image in path
    image_file = os.path.join(IMAGE_ROOT, file_name)
    shutil.copyfile(image_file, os.path.join(image_dir, file_name))

def save_mask(im, an):
    # convert coco to masks
    mask = annotation2binarymask(an, im['width'], im['height'])
    file_name = im['file_name']

    # create mask dir if doesn't exist
    mask_dir = os.path.join(MASK_ROOT, file_name[:-4], 'masks')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # save mask in path
    path = os.path.join(mask_dir, str(an['id']) + '.png')
    mask = Image.fromarray(mask * 255)
    mask.save(path, "PNG")


def main(jsonf):
    images, annotations = load_images(jsonf)

    j = 0
    for i, im in enumerate(images):
        save_image(im)
        while im['id'] == annotations[j]['image_id']:
            save_mask(im, annotations[j])
            j += 1

if __name__ == '__main__':
    main('/home/ajay/ccbr/dataset/cocom.json')