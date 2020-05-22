# import the necessary packages
import cv2
from time import time, sleep
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate
from utils import load_coco_names, letter_box_image, non_max_suppression, draw_boxes
import json
import random
import numpy as np

class VideoCamera(object):
    def __init__(self, filename, camera_id, model, label, params, prob_threshold=0.5, iou_threshold=0.5):
        with open(params, "r") as readFile:
            self.params = json.load(readFile)
        
        if filename:
            self.video = cv2.VideoCapture(filename)
        else:
            #self.video = cv2.VideoCapture(camera_id + cv2.CAP_DSHOW)
            self.video = cv2.VideoCapture(camera_id)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.params["input_w"])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params["input_h"])
    
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        
        # Initialize model
        self.interpreter = tflite.Interpreter(model_path=model, experimental_delegates=[load_delegate("libedgetpu.so.1.0")])
        self.interpreter.allocate_tensors() 
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()    
        # Dequantization
        self.scale, self.zero_point = self.input_details[0]['quantization']
        self.scale1, self.zero_point1 = self.output_details[0]['quantization']    
        self.scale2, self.zero_point2 = self.output_details[1]['quantization']
        
        self.classes = load_coco_names(label)
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(self.classes))]
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        
    def __del__(self):
        #releasing camera
        self.video.release()
    
    def detect(self, frame):
        t0 = time()
        
        frame = letter_box_image(frame, (self.params["input_w"], self.params["input_h"]), 128)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = np.expand_dims(img, axis=0).astype(np.float32)
        img = img[np.newaxis,...].astype(np.float32)
        img = np.uint8(img / self.scale + self.zero_point)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        y1 = self.interpreter.get_tensor(self.output_details[0]['index'])
        y2 = self.interpreter.get_tensor(self.output_details[1]['index'])
        y1 = self.scale1 * (np.float32(y1) - self.zero_point1)
        y2 = self.scale2 * (np.float32(y2) - self.zero_point2)
        
        detected_boxes = [y1, y2]
        filtered_boxes = non_max_suppression(detected_boxes,
                                            self.params,
                                            confidence_threshold=self.prob_threshold,
                                            iou_threshold=self.iou_threshold)
        draw_boxes(filtered_boxes, frame, self.classes, (self.params["input_w"], self.params["input_h"]), self.colors, True)
        
        inf_time = time() - t0
        fps = 1./inf_time
        
        cv2.putText(frame, "FPS: {:.1f}".format(fps),
                    (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        return frame
    
    def get_frame(self):
        t0 = time()
        #extracting frames
        ret, frame = self.video.read()
        if ret:
            frame = self.detect(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = self.video.read()
            frame = self.detect(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
        t1 = 1/self.fps - time() + t0
        if t1 > 0: sleep(t1)
        return jpeg.tobytes()