import numpy as np
from tflite_utils import letter_box_image
from argparse import ArgumentParser
import json
import cv2

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to a image file.",
                        required=True, type=str)
    parser.add_argument("-s", "--size", help="Size of input image", default=1024, type=int)    
    parser.add_argument("-o", "--output", help="Path to output image.", default=None, type=str)
    return parser

if __name__ == "__main__":
    
    args = build_argparser().parse_args()
    input_size = args.size
    input_file = args.input
    output_file = args.output
    
    cap = cv2.VideoCapture(input_file)
    
    encode = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file, encode, fps, (input_size, input_size), True)
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        
        if ret:
            
            frame = letter_box_image(frame, (input_size, input_size), 0)
            out.write(frame)
        
        else:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()