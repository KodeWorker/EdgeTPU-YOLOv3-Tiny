set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

MODEL=./models/0515/pen_demo.pb
OUTPUT=./models/0515/pen_demo.tflite
PARAMS=./inputs/pen/params.json

python convert_frozen_model_to_tflite.py \
-m $MODEL -o $OUTPUT \
--params $PARAMS