LABEL=./labels/pen.labels
MODEL=./models/0515/pen_demo_edgetpu.tflite
INPUT=./inputs/pen/demo_input_prep.mp4
OUTPUT=./outputs/pen/demo_output_simple.mp4
PARAMS=./inputs/pen/params.json
PROB_THRESHOLD=0.5
IOU_THRESHOLD=0.5

python3 demo_frame_execution_time_tinker.py \
-m $MODEL -I $INPUT -O $OUTPUT -l $LABEL \
-pt $PROB_THRESHOLD -iou $IOU_THRESHOLD \
--params $PARAMS