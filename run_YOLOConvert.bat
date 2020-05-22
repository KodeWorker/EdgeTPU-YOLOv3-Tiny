:: parameter
set yolo_class_names_file=Dockerfile/pen_demo.names
set yolo_weights_file=Dockerfile/pen_demo.weights
set pb_file=Dockerfile/pen_demo.pb
set model_size=384

echo Conver *.weights to *.pb 
::: Step1: Transform yolo-format(.weights) to tensorflow-format(.pb)
call python YOLOConvert/convert_weights_pb.py --class_names %yolo_class_names_file% --data_format NHWC --weights_file %yolo_weights_file% --output_graph %pb_file% --tiny --size %model_size% 