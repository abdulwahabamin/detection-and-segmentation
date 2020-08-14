from typing import Optional
import os

import numpy as np
import cv2
from PIL import Image

from OD_model.yolo import YOLO
from SG_model.segmentation import predict_
import utils

yolo = YOLO()


def predict_segmentation(base_path,folders, savedir):
    for folder in folders:
        images = os.listdir(os.path.join(base_path, folder, 'JPEGImages'))
        images = [k for k in images if '.jpg' or '.png' in k]
        if not os.path.exists(os.path.join(savedir, folder)):
            os.mkdir(os.path.join(savedir, folder))
            os.mkdir(os.path.join(savedir, folder, 'text'))
            os.mkdir(os.path.join(savedir, folder, 'Images'))

        for image in images:
            img = os.path.join(base_path, folder, 'JPEGImages', image)
            print('testing image ' + img + '\n')
            save_img, _, _, annot = predict_(img, save=True)
            im = image.split('.')[0]
            f = open(os.path.join(savedir, folder, 'text', im + '.txt'), 'w+')
            cv2.imwrite(os.path.join(savedir, folder, 'Images',im + '.jpg'), save_img)
            print(len(annot))
            for annotation in annot:
                f.write(annotation + '\n')
            f.close()


def predict(base_path, folders, savedir):
    for folder in folders:
        images = os.listdir(os.path.join(base_path, folder, 'JPEGImages'))
        images = [k for k in images if '.jpg' or '.png' in k]
        if not os.path.exists(os.path.join(savedir, folder)):
            os.mkdir(os.path.join(savedir, folder))

        for image in images:
            img = os.path.join(base_path, folder, 'JPEGImages', image)
            print('testing image ' + img + '\n')
            annot = combine_predctions_non_max(img)
            im = image.split('.')[0]
            f = open(os.path.join(savedir, folder, im + '.txt'), 'w+')
            print(len(annot))
            for annotation in annot:
                f.write(annotation + '\n')
            f.close()


def predict_percentage(trash_percentage, base_path, folders, savedir):
    for folder in folders:
        images = os.listdir(os.path.join(base_path, folder, 'JPEGImages'))
        images = [k for k in images if '.jpg' or '.png' in k]
        if not os.path.exists(os.path.join(savedir, folder)):
            os.mkdir(os.path.join(savedir, folder))

        for image in images:
            img = os.path.join(base_path, folder, 'JPEGImages', image)
            print('testing image ' + img + '\n')
            annot = combine_predctions_percentage(img, trash_threshold=trash_percentage)
            im = image.split('.')[0]
            f = open(os.path.join(savedir, folder, im + '.txt'), 'w+')
            print(len(annot))
            for annotation in annot:
                f.write(annotation + '\n')
            f.close()


def combine_predctions_non_max(image_path: str, confidence_threshold: Optional[float] = 0.3,
                       overlap_threshold: Optional[float] = 0.45, show_image: Optional[bool] = False):
    image = cv2.imread(image_path)
    od_image = Image.open(image_path)


    od_boxes = yolo.detect_image(od_image)
    _, _, sg_boxes_rescaled = predict_(image_path)

    all_boxes = list()
    all_boxes.extend(od_boxes)
    all_boxes.extend(sg_boxes_rescaled)

    confidence_scores, boxes = utils.non_max_suppression(all_boxes, overlap_threshold)

    predictions = list()

    for confidence, box_coordinates in zip(confidence_scores, boxes):
        if confidence > confidence_threshold:
            predictions.append('garbage' + ' ' + str(confidence) + " " + str(box_coordinates[0]) + ' ' +
                               str(box_coordinates[1]) + ' ' + str(box_coordinates[2]) + ' ' + str(box_coordinates[3]))

            cv2.rectangle(image, (box_coordinates[0], box_coordinates[1]), (box_coordinates[2],
                                                                            box_coordinates[3]),
                          color=(0, 0, 255), thickness=2)

    if show_image:
        cv2.imshow('filtered', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return predictions


def combine_predctions_percentage(image_path: str, trash_threshold: Optional[float] = 0.1,
                                  output_text_format: Optional[bool] = True, overlap_threshold: Optional[float] = 0.45,
                                  show_image: Optional[bool] = False):
    image = cv2.imread(image_path)
    od_image = Image.open(image_path)

    od_boxes_original = yolo.detect_image(od_image)
    if len(od_boxes_original) == 0:
        return []

    # Get  original boxes without rescaling
    trash_masks, sg_boxes_original, sg_boxes_rescaled = predict_(image_path)

    if len(trash_masks) == 0:
        return []

    # scaled down to 256 256 image size
    confid_od, od_boxes_rescaled = utils.rescale_boxes(trash_masks[0], image, od_boxes_original)

    # Calculate overlap and filter boxes

    # 1. Get rid of confidence scores from outputs
    sg_boxes = np.asarray(sg_boxes_original)[:, 1:]
    od_boxes_rescaled = np.asarray(od_boxes_rescaled)

    # 2. Calculate overlap for each od box with all segmentation boxes and filter out corresponding boxes

    # indexes_to_drop = list()
    predictions = list()
    for index, box in enumerate(od_boxes_rescaled):
        od_x = box[0]
        od_y = box[1]
        od_xx = box[2]
        od_yy = box[3]

        seg_x = sg_boxes[:, 0]
        seg_y = sg_boxes[:, 1]
        seg_xx = sg_boxes[:, 2]
        seg_yy = sg_boxes[:, 3]

        # Select values for overlap area calculation
        x_left = np.maximum(od_x, seg_x)
        y_top = np.maximum(od_y, seg_y)
        x_right = np.minimum(od_xx, seg_xx)
        y_bottom = np.minimum(od_yy, seg_yy)

        # filter out areas that don't overlap
        x_filter = x_right > x_left
        y_filter = y_bottom > y_top
        filter = x_filter * y_filter

        # check if there are intersecting areas and calculate iou if there are
        overlap_count = np.sum(filter)
        # if overlap_count == 0:
        #     indexes_to_drop.append(index)
        if overlap_count > 0:
            w = np.maximum(0, x_right - x_left + 1)
            h = np.maximum(0, y_bottom - y_top + 1)

            intersection_area = w * h
            od_area = (od_xx - od_x + 1) * (od_yy - od_y + 1)
            sg_area = (seg_xx - seg_x + 1) * (seg_yy - seg_y + 1)
            filter_indexes = np.nonzero(filter)[0]
            for id in filter_indexes:
                iou = intersection_area[id] / (od_area + sg_area[id] - intersection_area[id])
                # check iou and then
                if iou > overlap_threshold:
                    trash_ratio = od_area / np.count_nonzero(trash_masks[id])
                    if trash_ratio > trash_threshold:
                        if output_text_format:
                            box_coordinates = utils.rescale_box(image, trash_masks[0], box)
                            # box_coordinates = (np.asarray(sg_boxes_rescaled)[:, 1:])[id]
                            predictions.append('garbage' + ' ' + str(trash_ratio) + " " + str(box_coordinates[0]) + ' '
                                               + str(box_coordinates[1]) + ' ' + str(box_coordinates[2]) + ' ' +
                                               str(box_coordinates[3]))

                            cv2.rectangle(image, (box_coordinates[0], box_coordinates[1]), (box_coordinates[2],
                                                                                            box_coordinates[3]),
                                          color=(0, 0, 255), thickness=2)
    if show_image:
        cv2.imshow('filtered', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predictions


if __name__ == '__main__':
    folders = ['VOC_Test_Easy', 'VOC_Test_Hard']
    base_path = '/Ted/datasets/Garbage'
    main_savedir = '/Ted/models/results/od_seg_trash_percentage'

    if not os.path.exists(main_savedir):
        os.mkdir(main_savedir)

    predict_segmentation(base_path, folders, main_savedir)

    # for thresh in np.arange(0, 1, 0.05):
    #     save_dir = os.path.join('/Ted/models/results/od_seg_trash_percentage', str(thresh))
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     predict_percentage(thresh, base_path, folders, save_dir)

    # print(combine_predctions_percentage('test.jpg', output_text_format=True, show_image=True))
    # print(combine_predctions_non_max('test.jpg'))