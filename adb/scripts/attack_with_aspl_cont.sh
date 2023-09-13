export EXPERIMENT_NAME="ASPL"
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export CLEAN_TRAIN_DIR="db_dataset/$ID/set_A" 
export CLEAN_ADV_DIR="db_dataset/$ID/set_B"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/adversarial"
export CLASS_DIR="db_dataset/class_person"

mkdir -p $OUTPUT_DIR
cp -r $CLEAN_TRAIN_DIR/ $OUTPUT_DIR/clean_set
cp -r $CLEAN_ADV_DIR/ $OUTPUT_DIR/before_addding_noise_set

# ------------------------- Train DreamBooth on perturbed examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/dreambooth"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=fp16 \
  --prior_generation_precision=fp16 \
  --sample_batch_size=8 \
  --train_text_encoder \
