# -*- coding: utf-8 -*-
import numpy as np
import cv2
from math import exp, sqrt

def scale_bbox(x, y, h, w, class_id, confidence):
    xmin = int((x - w / 2))
    ymin = int((y - h / 2))
    xmax = int(xmin + w)
    ymax = int(ymin + h)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def non_max_suppression(predictions_with_boxes, params, confidence_threshold=0.5, iou_threshold=0.5):

    result = {}
    objects = list()
    input_w = params["input_w"]
    input_h = params["input_h"]
    num_anchors = int(len(params["anchors"])/4)
    num_classes = params["classes"]
    num_coords = params["coords"]
    num_bbox_attrs = num_classes+num_coords+1
    anchors = params["anchors"]
    
    sides = [p.shape[1] for p in predictions_with_boxes]
    grid = [(out_blob, side, anchor_offset, n) for out_blob, side, anchor_offset in zip(predictions_with_boxes, sides, (6, 0)) for n in range(num_anchors)]
    for out_blob, side, anchor_offset, n in grid:
        index = np.argwhere(sigmoid(out_blob[0,:,:,n*num_bbox_attrs+num_coords]) >= confidence_threshold)
        for row, col in index:
            for j in range(num_classes):
                confidence = sigmoid(out_blob[0,row,col,n*num_bbox_attrs+num_coords+j+1])
                if confidence < confidence_threshold:
                    continue
                x = (col + sigmoid(out_blob[0,row,col,n*num_bbox_attrs+0])) / side * input_w
                y = (row + sigmoid(out_blob[0,row,col,n*num_bbox_attrs+1])) / side * input_h
                w = np.exp(out_blob[0,row,col,n*num_bbox_attrs+2]) * anchors[anchor_offset + 2 * n]
                h = np.exp(out_blob[0,row,col,n*num_bbox_attrs+3]) * anchors[anchor_offset + 2 * n + 1]
                objects.append(scale_bbox(x,y,h,w,j,confidence))
        
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    objects = [obj for obj in objects if obj['confidence'] >= confidence_threshold]
    
    for object in objects:
        box = np.array([object["xmin"], object["ymin"], object["xmax"], object["ymax"]])
        score = object["confidence"]
        cls = object["class_id"]
        if cls not in result:
            result[cls] = []
        result[cls].append((box, score))
    return result

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def draw_boxes(boxes, img, cls_names, size, colors, is_letter_box_image, is_class_name_only=True):
    iw, ih = img.shape[0:2][::-1]
    w, h = size
    w_scale, h_scale = iw/w, ih/h
    for cls, bboxs in boxes.items():
        color = colors[cls]
        for box, score_ in bboxs:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1*h_scale), int(y1*w_scale)), (int(x2*h_scale), int(y2*w_scale)), color, 2)
            if is_class_name_only:
                cv2.putText(img, "{}".format(cls_names[cls]),
                        (int(x1*h_scale), int(y1*h_scale)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "{} {:.2f}%".format(cls_names[cls], score_ * 100),
                        (int(x1*h_scale), int(y1*h_scale)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

def letter_box_image(image, size, fill=128):
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(fill)
    dx = (w-iw)//2
    dy = (h-ih)//2
    new_image[dy:dy+ih, dx:dx+iw,:] = image
    return new_image