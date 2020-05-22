set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

LABEL=./labels/pen.labels
MODEL=./models/0512/pen_demo_simple.tflite
INPUT=./inputs/pen/demo_input_prep.mp4
OUTPUT=./outputs/pen/demo_output_simple.mp4
PARAMS=./inputs/pen/params.json
PROB_THRESHOLD=0.5
IOU_THRESHOLD=0.5

python demo_frame_execution_time.py \
-m $MODEL -I $INPUT -O $OUTPUT -l $LABEL \
-pt $PROB_THRESHOLD -iou $IOU_THRESHOLD \
--params $PARAMS