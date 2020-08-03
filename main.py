from typing import Optional
import os

import numpy as np
import cv2
from PIL import Image

from OD_model.yolo import YOLO
from SG_model.segmentation import predict_
from utils import non_max_suppression


yolo = YOLO()


def predict(base_path, folders, savedir):
    for folder in folders:
        images = os.listdir(os.path.join(base_path, folder, 'JPEGImages'))
        images = [k for k in images if '.jpg' or '.png' in k]
        if not os.path.exists(os.path.join(savedir, folder)):
            os.mkdir(os.path.join(savedir, folder))

        for image in images:
            img = os.path.join(base_path, folder, 'JPEGImages', image)
            print('testing image ' + img + '\n')
            annot = combine_predctions(img)
            im = image.split('.')[0]
            f = open(os.path.join(savedir, folder, im + '.txt'), 'w+')
            print(len(annot))
            for annotation in annot:
                f.write(annotation + '\n')
            f.close()


def combine_predctions(image_path: str, confidence_threshold: Optional[float] = 0.3,
                       overlap_threshold: Optional[float] = 0.45):
    # image = cv2.imread(image_path)
    od_image = Image.open(image_path)

    od_boxes = yolo.detect_image(od_image)
    sg_boxes = predict_(image_path)

    all_boxes = list()
    all_boxes.extend(od_boxes)
    all_boxes.extend(sg_boxes)
    all_boxes = np.array(all_boxes)

    confidence_scores, boxes = non_max_suppression(all_boxes, overlap_threshold)

    boxes = list()

    for confidence, box_coordinates in zip(confidence_scores, boxes):
        if confidence > confidence_threshold:
            boxes.append(str(confidence) + ' ' + str(box_coordinates[0]) + ' ' + str(box_coordinates[1]) + ' ' +
                         str(box_coordinates[2]) + ' ' + str(box_coordinates[3]))

    #         cv2.rectangle(image, (box_coordinates[0], box_coordinates[1]), (box_coordinates[2],
    #                                                                         box_coordinates[3]),
    #                       color=(0, 0, 255), thickness=2)
    #
    # cv2.imshow('filtered', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return boxes


if __name__ == '__main__':
    folders = ['VOC_Test_Easy', 'VOC_Test_Hard']
    base_path = '/Ted/datasets/Garbage'
    save_dir = '/Ted/models/results/od_seg'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    predict(base_path, folders, save_dir)
