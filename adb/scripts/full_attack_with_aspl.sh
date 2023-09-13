DB_DATASET_PATH='db_dataset/'

for folder in '$DB_DATASET_PATH/*'; do
    if [ -d '$folder']; then
        folder_name=$(basename "$folder")
        
        # Update the variables
        EXPERIMENT_NAME="ASPL"
        MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
        CLEAN_TRAIN_DIR="db_dataset/$folder_name/set_A"
        CLEAN_ADV_DIR="db_dataset/$folder_name/set_B"
        OUTPUT_DIR="outputs/$EXPERIMENT_NAME/${folder_name}_ADVERSARIAL"
        CLASS_DIR="db_dataset/class-person"
        
        if [ -d "outputs/{$folder_name}_ADVERSARIAL/noise-ckpt/50" ]; then
            echo "Skipping set $folder_name because it is already trained."
            continue
        fi

        # Perform actions with the updated variables
        # For example, you can run your script or commands here
        # ------------------------- Train ASPL on set B -------------------------
        mkdir -p $OUTPUT_DIR
        cp -r $CLEAN_TRAIN_DIR $OUTPUT_DIR/image_clean
        cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

        accelerate launch --main_process_port 29051 attacks/aspl.py \
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
        --checkpointing_iterations=10 \
        --learning_rate=5e-7 \
        --pgd_alpha=5e-3 \
        --pgd_eps=5e-2 \
        #--train_text_encoder \

        # ------------------------- Train DreamBooth on perturbed examples -------------------------
        export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
        export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/{$folder_name}_DREAMBOOTH"

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
        --checkpointing_steps=500 \
        --center_crop \
        --mixed_precision=fp16 \
        --prior_generation_precision=fp16 \
        --sample_batch_size=8 \
        --train_text_encoder \

    fi
done


