from flask import Flask, render_template, Response
from detector import VideoCamera
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--host", help="IP host of stream server", required=True, type=str)
    parser.add_argument("--port", help="Port of stream server", required=True, type=str)
    parser.add_argument("-m", "--model", help="Path to a .pb file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a image file.", default=None, type=str)
    parser.add_argument("-c", "--camera_id", help="Camera ID.", type=int)
    parser.add_argument("--params", help="Path to a .json file with parameters", required=True, default=None, type=str)
    parser.add_argument("-l", "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-iout", "--iou_threshold", help="Intersection over union threshold for overlapping detections"
                                                        " filtering", default=0.4, type=float)
    return parser

app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
    
def generate(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
               
@app.route('/video_feed')
def video_feed():
    return Response(generate(VideoCamera(filename=filename,
                                        camera_id=camera_id,
                                        model=model,
                                        label=label,
                                        params=params,
                                        prob_threshold=prob_threshold,
                                        iou_threshold=prob_threshold)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
if __name__ == '__main__':
    args = build_argparser().parse_args()
    
    host=args.host
    port=args.port
    
    global filename
    global camera_id
    global model
    global label
    global params
    global prob_threshold
    global iou_threshold
    
    filename=args.input
    camera_id=args.camera_id
    model=args.model
    label=args.labels
    params=args.params
    prob_threshold=args.prob_threshold
    iou_threshold=args.iou_threshold
    
    # defining server ip address and port
    app.run(host=host,port=port, debug=True)
    