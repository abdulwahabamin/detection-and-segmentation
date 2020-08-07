from typing import List, Tuple, Union, Optional

import numpy as np


def rescale_box(original_image: np.ndarray, resized_image: np.ndarray, box_coordinates: List) -> List:
    original_height, original_width, _ = original_image.shape
    resized_height, resized_width, _ = resized_image.shape
    height_ratio, width_ratio = original_height/resized_height, original_width/resized_width
    x = box_coordinates[0]
    y = box_coordinates[1]
    xx = box_coordinates[2]
    yy = box_coordinates[3]
    box_width = abs(xx - x) * width_ratio
    box_height = abs(yy - y) * height_ratio
    box_start_x = int(x * width_ratio)
    box_start_y = int(y * height_ratio)
    box_end_x = int(box_start_x + box_width)
    box_end_y = int(box_start_y + box_height)
    rescaled_box_coordinates = [box_start_x, box_start_y, box_end_x, box_end_y]
    return rescaled_box_coordinates


def rescale_boxes(original_image: np.ndarray, resized_image: np.ndarray, boxes: List):
    coord = list()
    confidence = list()
    for box in boxes:
        coord.append(rescale_box(original_image, resized_image, box[1:]))
        confidence.append(box[0])

    return confidence, coord


def non_max_suppression(boxes: List, overlapThresh: float = 0.45):
    boxes = np.asarray(boxes)
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type for rectangle coordinates
    confidence_scores = boxes[pick][:, 0]
    boxes = boxes[pick][:, 1:]
    return confidence_scores, boxes.astype("int")



