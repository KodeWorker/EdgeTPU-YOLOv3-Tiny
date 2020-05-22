if [%1]==[] goto exit

@echo Do file transfer to edge (%1)
cd dockerfile_transfer

docker build -f DockerfileBASE -t transfer/base .
docker rmi transfer/ssh
docker build -f DockerfileSSH -t transfer/ssh .
docker run --rm transfer/ssh scp -i /key/id_rsa /params/pen.json mendel@%1:/home/mendel/tflite_conversion/params
docker run --rm transfer/ssh scp -i /key/id_rsa /labels/pen.labels mendel@%1:/home/mendel/tflite_conversion/labels
docker run --rm transfer/ssh scp -i /key/id_rsa /models/pen_demo_edgetpu.tflite mendel@%1:/home/mendel/tflite_conversion/models
goto :eof

:exit
@echo Please pass edge host IP.
goto :eof


