@echo off
IF "%~1"=="" GOTO :ENTERIP
IF "%~2"=="--skip" GOTO :DOCVT

@echo "Build docker images"
:: Pull images from tensorflow
docker pull tensorflow/tensorflow:2.2.0
docker pull tensorflow/tensorflow:1.11.0-py3
:: Build comvert & compile environment
docker build -f Dockerfile/DockerfileConverter -t edgetpu/convert .
docker build -f Dockerfile/DockerfileCompiler -t edgetpu/compile .

:DOCVT
@echo "Convert and transfer model"
:: Convert YOLOv3-tiny model 
docker rmi edgetpu/cvt
docker build --no-cache -f Dockerfile/DockerfileCVT -t edgetpu/cvt .
docker run -d --name temp edgetpu/cvt
docker cp temp:/models/pen_demo.pb Dockerfile/pen_demo.pb
docker container stop temp
docker rm temp

:: Compile tensorflow model for edgetpu
docker rmi edgetpu/cmp
docker build --no-cache -f Dockerfile/DockerfileCMP -t edgetpu/cmp .
docker run -d --name temp edgetpu/cmp
docker cp temp:/models/pen_demo_edgetpu.tflite Dockerfile/pen_demo_edgetpu.tflite
docker cp temp:/models/pen_demo.labels Dockerfile/pen_demo.labels
docker cp temp:/models/pen_demo.json Dockerfile/pen_demo.json
docker container stop temp
docker rm temp

:: Transfer model through SSH
docker run -it --rm edgetpu/cmp scp -i /key/id_rsa /models/pen_demo_edgetpu.tflite mendel@%1:/home/mendel/tflite_conversion/models
docker run -it --rm edgetpu/cmp scp -i /key/id_rsa /models/pen_demo.labels mendel@%1:/home/mendel/tflite_conversion/labels
docker run -it --rm edgetpu/cmp scp -i /key/id_rsa /models/pen_demo.json mendel@%1:/home/mendel/tflite_conversion/params

GOTO :EOF

:ENTERIP
@echo "Please ENTER IP address of TinkerEdgeT."