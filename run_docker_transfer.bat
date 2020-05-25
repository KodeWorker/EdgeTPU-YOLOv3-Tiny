if [%1]==[] goto exit

@echo Do file transfer to edge (%1)
cd dockerfile_transfer

docker run --rm edgetpu/cmp scp -i /key/id_rsa /params/pen.json mendel@%1:/home/mendel/tflite_conversion/params
docker run --rm edgetpu/cmp scp -i /key/id_rsa /labels/pen.labels mendel@%1:/home/mendel/tflite_conversion/labels
docker run --rm edgetpu/cmp scp -i /key/id_rsa /models/pen_demo_edgetpu.tflite mendel@%1:/home/mendel/tflite_conversion/models
goto :eof

:exit
@echo Please pass edge host IP.


