export EXPERIMENT_NAME="ASPL"
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export CLASS_DIR="db_dataset/class-person"

# Specify the directory
dataset_dir="db_dataset"

# Use a for loop to iterate over all folders in the directory
for subset_dir in "$dataset_dir"/*
do
    # Check if it's a directory
    if [ -d "$subset_dir" ]; then
        subset_dir=$(basename $subset_dir)
        echo "Training set $subset_dir"
        # Your code here
        export ID=$subset_dir
        export CLEAN_TRAIN_DIR="db_dataset/$ID/set_A" 
        export CLEAN_ADV_DIR="db_dataset/$ID/set_B"
        export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/adversarial"

        # ------------------------- Train ASPL on set B -------------------------
        mkdir -p $OUTPUT_DIR
        cp -r $CLEAN_TRAIN_DIR/ $OUTPUT_DIR/image_clean
        cp -r $CLEAN_ADV_DIR/ $OUTPUT_DIR/image_before_addding_noise

        accelerate launch --main_process_port 29052 attacks/aspl.py \
          --pretrained_model_name_or_path=$MODEL_PATH  \
          --enable_xformers_memory_efficient_attention \
          --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
          --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
          --instance_prompt="a photo of sks person" \
          --class_data_dir=$CLASS_DIR \
          --num_class_images=200 \
          --class_prompt="a photo of person" \
          --output_dir=$OUTPUT_DIR \
          --center_crop \
          --with_prior_preservation \
          --prior_loss_weight=1.0 \
          --resolution=512 \
          --train_batch_size=1 \
          --max_train_steps=50 \
          --max_f_train_steps=3 \
          --max_adv_train_steps=6 \
          --checkpointing_iterations=50 \
          --learning_rate=5e-7 \
          --pgd_alpha=5e-3 \
          --pgd_eps=5e-2 \
          --train_text_encoder \

        # ------------------------- Train DreamBooth on perturbed examples -------------------------
        export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/adversarial"
        export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
        export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/dreambooth"

        accelerate launch ./src/train_dreambooth.py \
          --id=$id \
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

    fi
done


