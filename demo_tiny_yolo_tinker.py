import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate
from tflite_utils import letter_box_image, load_coco_names, non_max_suppression, draw_boxes
import cv2
from argparse import ArgumentParser
from time import time
import json
import random

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to a .tflite file with a trained model.", required=True, type=str)
    parser.add_argument("-I", "--input", help="Path to a image file.",
                        required=True, type=str)
    parser.add_argument("--params", help="Path to a .json file with parameters", required=True, default=None, type=str)
    parser.add_argument("-l", "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-iout", "--iou_threshold", help="Intersection over union threshold for overlapping detections"
                                                         " filtering", default=0.4, type=float)
    parser.add_argument("-O", "--output", help="Path to output image.", default=None, type=str)
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    tflite_model_path = args.model
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    input_img = args.input
    output_img = args.output
    class_names = args.labels
    params_ = args.params
    
    with open(params_, "r") as readFile:
        params = json.load(readFile)
    
    img = cv2.imread(input_img)
    resized_img = letter_box_image(img, (params["input_w"], params["input_h"]), 128)
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    classes = load_coco_names(class_names)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(classes))]
    
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=tflite_model_path, experimental_delegates=[load_delegate("libedgetpu.so.1.0")])
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Dequantization
    scale, zero_point = input_details[0]['quantization']
    img = np.uint8(img / scale + zero_point)
    
    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    y1 = interpreter.get_tensor(output_details[0]['index'])
    y2 = interpreter.get_tensor(output_details[1]['index'])
    # Dequantization
    scale1, zero_point1 = output_details[0]['quantization']
    y1 = scale1 * (np.float32(y1) - zero_point1)
    scale2, zero_point2 = output_details[1]['quantization']
    y2 = scale2 * (np.float32(y2) - zero_point2)
    
    detected_boxes = [y1, y2]
    
    filtered_boxes = non_max_suppression(detected_boxes,
                                         params,
                                         confidence_threshold=prob_threshold,
                                         iou_threshold=iou_threshold)
    draw_boxes(filtered_boxes, resized_img, classes, (params["input_w"], params["input_h"]), colors, True)
    
    if output_img:
        cv2.imwrite(output_img, resized_img)
