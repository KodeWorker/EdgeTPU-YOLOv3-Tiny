import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow
if tensorflow.__version__[0] == "2":
    import tensorflow.compat.v1 as tf
elif tensorflow.__version__[0] == "1":
    import tensorflow as tf

from tensorflow.python.platform import gfile
from argparse import ArgumentParser
import numpy as np
from tflite_utils import letter_box_image
import json

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to a .pb file with a trained model.", required=True, type=str)
    parser.add_argument("-o", "--output", help="Path to a .tflite file with a converted model.", default=None, type=str)
    parser.add_argument("--params", help="Path to a .json file with parameters", required=True, default=None, type=str)
    return parser

def representative_dataset_gen():
    
    num_calibration_steps = 10
    for _ in range(num_calibration_steps):
        # Get sample input data as a numpy array in a method of your choosing.
        input = np.random.randint(0, 255, size=(1,input_size[0],input_size[1],3)).astype(np.float32)
        yield [input]
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    GRAPH_PB_PATH = args.model
    OUTPUT_TFLITE = args.output
    params_ = args.params
    
    with open(params_, "r") as readFile:
        params = json.load(readFile)
    
    global input_size
    input_size = (params["input_h"], params["input_w"])
    
    tf.enable_eager_execution()
    
    with tf.Session() as sess:
        print("LOAD GRAPH")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        x = sess.graph.get_tensor_by_name("inputs:0")
        y1 = sess.graph.get_tensor_by_name("detector/yolo-v3-tiny/Conv_9/BiasAdd:0")
        y2 = sess.graph.get_tensor_by_name("detector/yolo-v3-tiny/Conv_12/BiasAdd:0")
        
        # for accelerator
        converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y1, y2])
        
        # Quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen        
        converter.target_spec.supportrf_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_buffer = converter.convert()
        tf.gfile.GFile(OUTPUT_TFLITE, "wb").write(tflite_buffer)

    print("CONVERSION COMPLETE")