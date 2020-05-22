set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

MODEL=./models/0515/pen_demo.tflite
INPUT=./inputs/pen/test_create_12.png
OUTPUT=./outputs/pen/test_create_12_tflite.png
PARAMS=./inputs/pen/params.json
LABEL=./labels/pen.labels
PROB_THRESHOLD=0.5
IOU_THRESHOLD=0.5


python demo_tiny_yolo_tflite.py \
-m $MODEL -I $INPUT -O $OUTPUT -l $LABEL \
-pt $PROB_THRESHOLD -iou $IOU_THRESHOLD \
--params $PARAMS