set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

MODEL=./models/0515/pen_demo.pb
INPUT=./inputs/pen/test_create_12.png
OUTPUT=./outputs/pen/test_create_12.png
PARAMS=./inputs/pen/params.json
LABEL=./labels/pen.labels

#MODEL=./models/0422/dit_fx.pb
#INPUT=./inputs/fruit/test_create_1.png
#OUTPUT=./outputs/fruit/test_create_1m.png
#PARAMS=./inputs/fruit/params.json
#LABEL=./labels/fruit.labels

#MODEL=./models/0512/pen_demo_simple.pb
#INPUT=./inputs/pen/test_create_12.png
#OUTPUT=./outputs/pen/test_create_12.png
#PARAMS=./inputs/pen/params.json
#LABEL=./labels/pen.labels

PROB_THRESHOLD=0.5
IOU_THRESHOLD=0.5
GPU_MEMORY_FRACTION=1.0

python demo_tiny_yolo_tf.py \
-m $MODEL -I $INPUT -O $OUTPUT -l $LABEL \
-pt $PROB_THRESHOLD -iou $IOU_THRESHOLD \
-mf $GPU_MEMORY_FRACTION \
--params $PARAMS