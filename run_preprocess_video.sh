SIZE=384
INPUT=./inputs/pen/demo_input.mp4
OUTPUT=./inputs/pen/demo_input_prep.mp4

python3 preprocess_video.py \
-i $INPUT -o $OUTPUT -s $SIZE