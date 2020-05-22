import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import random
import tensorflow.compat.v1 as tf
from utils import letter_box_image, load_coco_names, get_boxes_and_inputs_pb, load_graph, non_max_suppression, draw_boxes
import cv2
from argparse import ArgumentParser
import json

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to a .pb file with a trained model.", required=True, type=str)
    parser.add_argument("-I", "--input", help="Path to a image file.",
                        required=True, type=str)
    parser.add_argument("--params", help="Path to a .json file with parameters", required=True, default=None, type=str)
    parser.add_argument("-l", "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-iout", "--iou_threshold", help="Intersection over union threshold for overlapping detections"
                                                         " filtering", default=0.4, type=float)
    parser.add_argument("-O", "--output", help="Path to output image.", default=None, type=str)
    parser.add_argument("-mf", "--gpu_memory_fraction", help="GPU configuration.", default=1.0, type=float)
    
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    frozen_model_path = args.model
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    gpu_memory_fraction = args.gpu_memory_fraction
    input_img = args.input
    output_img = args.output
    class_names = args.labels
    params_ = args.params
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )
    
    with open(params_, "r") as readFile:
        params = json.load(readFile)
        
    origin_img = cv2.imread(input_img)
    resized_img = letter_box_image(origin_img, (params["input_w"], params["input_h"]), 128)
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    classes = load_coco_names(class_names)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classes))]
    
    frozenGraph = load_graph(frozen_model_path)

    boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)    
    outputs = {}
    with tf.Session(graph=frozenGraph, config=config) as sess:
        for i in range(len(boxes)):
            outputs[boxes[i].name] = sess.run(boxes[i], feed_dict={inputs: [img]})
    detected_boxes = list(outputs.values())
    
    filtered_boxes = non_max_suppression(detected_boxes,
                                         params,
                                         confidence_threshold=prob_threshold,
                                         iou_threshold=iou_threshold)
    draw_boxes(filtered_boxes, origin_img, classes, (params["input_w"], params["input_h"]), colors, True)
    
    if output_img:
        cv2.imwrite(output_img, origin_img)