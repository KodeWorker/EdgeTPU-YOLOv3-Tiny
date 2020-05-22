import os
import configparser
import json
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to a yolo model *.cfg file.",
                        required=True, type=str)
    parser.add_argument("-l", "--label", help="Labels mapping *.names file.", default=None, type=str)
    parser.add_argument("-o", "--output", help="Output directory", default=None, type=str)
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    input = args.input
    label = args.label
    output = args.output
    
    params = {"classes":None, "coords": 4, "anchors": None, "input_w": None, "input_h": None}
    
    # label info extraction
    
    with open(label, "r") as readFile:
        labelLines = readFile.readlines()
    
    labelsFilename = os.path.basename(label.replace(".names", ".labels"))
    with open(os.path.join(output, labelsFilename), "w") as writeFile:
        for labelLine in labelLines:
            writeFile.write("{}".format(labelLine))
    
    # config info extraction
    config = configparser.ConfigParser(strict=False)
    config.read(input)
    
    params["classes"] = config.getint("yolo", "classes")
    params["input_w"] = config.getint("net", "width")
    params["input_h"] = config.getint("net", "height")
    anchors = config.get("yolo", "anchors")
    anchors = anchors.split(",")
    params["anchors"] = [int(anchor) for anchor in anchors]
    
    paramsFilename = os.path.basename(input.replace(".cfg", ".json"))
    with open(os.path.join(output, paramsFilename), 'w') as json_file:
        json.dump(params, json_file)