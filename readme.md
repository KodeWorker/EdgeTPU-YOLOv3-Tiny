# EdgeTPU-YOLOv3-Tiny

## Introduction
We build a docker workflow to compile and deploy YOLOv3-Tiny models for CoralDevBoard/TinkerEdgeT.
The inference results would shown on the website (http://\[EdgeTPU-IP\]:5000/).

### Firstly, we setup the CoralDevBoard/TinkerEdgeT for the first time.
1. Clone this repository to "/tflite_conversion"
2. Build OpenCV
3. Setup Flask and Startup service (systemd)
4. Generate public/private key pairs for SSH

**note:** You should have model/labels/params files for runing this web service.

### Secondely, we run the run_deploy_model.bat \[EdgeTPU-IP\] \[--skip\].
Before running this batch, we should have 4 files in Dockerfile/: private key, model.cfg, model.weights, and labels.names.
This batch file will build and convert YOLOv3-Tiny model to desired files for EdgeTPU.
The desired files (model/labels/params) would be generated in Dockerfile/.

**note:** You have to build YOLOv3-Tiny model using relu rather leaky-relu, because only relu activation funtion is supported for edgetpu_compiler.

## Project Structure
- Dockerfile/: Dockerfiles for building convert/compile environments
- YOLOConvert/: The code for converting YOLOv3-Tiny to tensorflow model
- stream/: Flask code for camera streaming demo
- .gitignore: Git ignore file
- convert_frozen_model_to_tflite.py: Convert frozen model (\*.pb) to quantized tflite model
- demo_frame_execution_time.py: Run video inference and show the execution time
- demo_frame_execution_time_tinker.py: Run video inference and show the execution time on edge device
- demo_tiny_yolo_tf.py: Run image inference on local PC using forzen model (\*.pb)
- demo_tiny_yolo_tflite.py: Run image inference on local PC using quantized tflite model (\*.tflite)
- demo_tiny_yolo_tinker.py: Run image inference on edge device using compiled tflite model (\*_edgetpu.tflite)
- generate_model_params.py: Generate \*.labels from \*.names, and \*.params from \*.cfg
- preprocess_video.py: Preprocess video for streaming demo
- readme.md: Read me file
- requirements.txt: PIP requirements for tensorflow2 (not for YOLOConvert)
- run_convert_frozen_model_to_tflite.sh: BASH file to run convert_frozen_model_to_tflite.py
- run_demo_frame_execution_time.sh: BASH file to run demo_frame_execution_time.py
- run_demo_frame_execution_time_tinker.sh: BASH file to run demo_frame_execution_time_tinker.py
- run_demo_tiny_yolo_tf.sh: BASH file to run demo_tiny_yolo_tf.py
- run_demo_tiny_yolo_tflite.sh: BASH file to run demo_tiny_yolo_tflite.py
- run_demo_tiny_yolo_tinker.sh: BASH file to run demo_tiny_yolo_tinker.py
- run_deploy_model.bat: BATCH file to deploy model
- run_docker_transfer.bat: BATCH file to transfer model using SSH
- run_preprocess_video.sh: BASH file to run preprocess_video.py
- run_stream_server.sh: BASH file to run stream/maing.py (Flask service)
- run_YOLOConvert.bat: BATCH file to convert YOLOv3-Tiny model to frozen model (\*.pb)
- utils.py: Utility code for local PC.
- tflite_utils.py: Utility code for edge device.

## References
- [Convert YOLOv3-Tiny to tensorflow model](https://github.com/mystic123/tensorflow-yolo-v3)
- [Run Tiny YOLO-v3 on Google's Edge TPU USB Accelerator](https://github.com/guichristmann/edge-tpu-tiny-yolo)
- [Installing OpenCV 4.0 on Google Coral Dev board](https://medium.com/@balaji_85683/installing-opencv-4-0-on-google-coral-dev-board-5c3a69d7f52f)
- [Coral DevBoardでOpenCVをビルドしてインストールする](https://qiita.com/rhene/items/4419ef08b85e697fe8c0)
- [OpenCV – Stream video to web browser/HTML page](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/)