MODEL=./models/0515/pen_demo_edgetpu.tflite
INPUT=./inputs/pen/test_create_2.png
OUTPUT=./outputs/pen/test_create_2_tinker.png
PARAMS=./inputs/pen/params.json
LABEL=./labels/pen.labels
PROB_THRESHOLD=0.5
IOU_THRESHOLD=0.5

python3 demo_tiny_yolo_tinker.py \
-m $MODEL -I $INPUT -O $OUTPUT -l $LABEL \
-pt $PROB_THRESHOLD -iou $IOU_THRESHOLD \
--params $PARAMS
