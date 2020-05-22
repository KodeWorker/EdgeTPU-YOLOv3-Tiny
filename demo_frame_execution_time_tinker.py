import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate
from tflite_utils import load_coco_names, letter_box_image, non_max_suppression, draw_boxes
from argparse import ArgumentParser
import json
import cv2
import time
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
    class_names = args.labels
    tflite_model_path = args.model
    input_file = args.input
    params_ = args.params
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    output_file = args.output
    
    output_file = "./outputs/pen/demo_output_simple.mp4"
    
    with open(params_, "r") as readFile:
        params = json.load(readFile)
    
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
    scale1, zero_point1 = output_details[0]['quantization']    
    scale2, zero_point2 = output_details[1]['quantization']
    
    cap = cv2.VideoCapture(input_file)
    encode = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file, encode, fps, (params["input_w"], params["input_h"]), True)
    
    avg_time = []
    avg_t1 = []
    avg_t2 = []
    avg_t3 = []
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        
        if ret:
            
            t0 = time.time()
            
            frame = letter_box_image(frame, (params["input_w"], params["input_h"]), 128)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #img = np.expand_dims(img, axis=0).astype(np.float32)
            img = img[np.newaxis,...].astype(np.float32)
            img = np.uint8(img / scale + zero_point)
            
            t1 = time.time() - t0
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            y1 = interpreter.get_tensor(output_details[0]['index'])
            y2 = interpreter.get_tensor(output_details[1]['index'])
            y1 = scale1 * (np.float32(y1) - zero_point1)
            y2 = scale2 * (np.float32(y2) - zero_point2)
            
            t2 = time.time() - t1 - t0
            
            detected_boxes = [y1, y2]
            filtered_boxes = non_max_suppression(detected_boxes,
                                                params,
                                                confidence_threshold=prob_threshold,
                                                iou_threshold=iou_threshold)
            draw_boxes(filtered_boxes, frame, classes, (params["input_w"], params["input_h"]), colors, True)
            
            t3 = time.time() - t2 - t1 -t0
            
            out.write(frame)
            avg_time.append(t1+t2+t3)
            avg_t1.append(t1)
            avg_t2.append(t2)
            avg_t3.append(t3)
            print("Avg. frame execution time: {:.4f} ({:.4f}, {:.4f}, {:.4f}) sec.".format(np.mean(avg_time), np.mean(avg_t1), np.mean(avg_t2), np.mean(avg_t3)), end="\r")
            
        else:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
