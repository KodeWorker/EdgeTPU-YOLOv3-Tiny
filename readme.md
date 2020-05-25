# EdgeTPU-YOLOv3-Tiny

## Introduction
We build a docker workflow to compile and deploy YOLOv3-Tiny models for CoralDevBoard/TinkerEdgeT.
The inference results would shown on the website (http://\[EdgeTPU-IP\]:5000/).

- Firstly, we setup the CoralDevBoard/TinkerEdgeT for the first time.
1. Clone this repository
2. Build OpenCV
3. Setup Flask and Startup service (systemd)
4. Generate public/private key pairs for SSH
**note:** you should have model/labels/params files for runing this web service.

- Secondely, we run the run_deploy_model.bat \[EdgeTPU-IP\] \[--skip\].
Before running this batch, we should have 4 files in Dockerfile/: private key, model.cfg, model.weights, and labels.names.
This batch file will build and convert YOLOv3-Tiny model to desired files for EdgeTPU.
The desired files (model/labels/params) would be generated in Dockerfile/.
**note:** you have to build YOLOv3-Tiny model using relu rather leaky-relu, because only relu activation funtion is supported for edgetpu_compiler.

## References
- [Convert YOLOv3-tiny to tensorflow model](https://github.com/mystic123/tensorflow-yolo-v3)
- [Run Tiny YOLO-v3 on Google's Edge TPU USB Accelerator](https://github.com/guichristmann/edge-tpu-tiny-yolo)
- [Installing OpenCV 4.0 on Google Coral Dev board](https://medium.com/@balaji_85683/installing-opencv-4-0-on-google-coral-dev-board-5c3a69d7f52f)
- [Coral DevBoardでOpenCVをビルドしてインストールする](https://qiita.com/rhene/items/4419ef08b85e697fe8c0)
- [OpenCV – Stream video to web browser/HTML page](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/)