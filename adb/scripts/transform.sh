export ID="124"
export TARGET_DIR="outputs/ASPL/$ID/adversarial/noise-ckpt/50"
export OUTPUT_DIR="outputs/ASPL/$ID/adversarial"
export CONFIG="config/transform.yaml"

python src/transform.py \
    --target_dir=$TARGET_DIR \
    --output_dir=$OUTPUT_DIR \
    --config=$CONFIG