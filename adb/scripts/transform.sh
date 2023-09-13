export ID=nam
export TARGET_DIR=data/$ID/set_A
export OUTPUT_DIR=outputs/$ID
export CONFIG=config/transform.yaml

python src/transform.py \
    --target_dir=$TARGET_DIR \
    --output_dir=$OUTPUT_DIR \
    --config=$CONFIG