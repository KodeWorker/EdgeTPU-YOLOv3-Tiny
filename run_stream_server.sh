HOST="192.168.1.137"
PORT="5000"
INPUT="./inputs/pen/demo_input_prep.mp4"
CAMERA_ID=2
MODEL="./models/pen_demo_edgetpu.tflite"
LABEL="./labels/pen_demo.labels"
PARAMS="./params/pen_demo.json"
PROB_THRESHOLD=0.5
IOT_THRESHOLD=0.5

python3 stream/main.py \
--host $HOST --port $PORT \
#-i $INPUT \
-c $CAMERA_ID \
-m $MODEL -l $LABEL --params $PARAMS \
-pt $PROB_THRESHOLD -iout $IOT_THRESHOLD