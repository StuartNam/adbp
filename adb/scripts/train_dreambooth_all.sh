dataset_dir="db_dataset"

for subset_dir in "$dataset_dir"/*
do
    if [ -d "$subset_dir" ]; then
        subset_dir=$(basename $subset_dir)
        echo "Training set $subset_dir"

        export ID=$subset_dir
        export EXPERIMENT_NAME='DB'
        export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
        export INSTANCE_DIR="db_dataset/$ID/set_A/"
        export CLASS_DIR="db_dataset/class-person"
        export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$ID/"

        accelerate launch ./src/train_dreambooth.py \
          --id=$ID \
          --pretrained_model_name_or_path=$MODEL_PATH  \
          --enable_xformers_memory_efficient_attention \
          --train_text_encoder \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DIR \
          --output_dir=$DREAMBOOTH_OUTPUT_DIR \
          --with_prior_preservation \
          --prior_loss_weight=1.0 \
          --instance_prompt="a photo of sks person" \
          --class_prompt="a photo of person" \
          --inference_prompt="a photo of sks person;a DSLR portrait of sks person" \
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
          --mixed_precision="fp16" \
          --prior_generation_precision="fp16" \
          --sample_batch_size=8 \
          --resume_from_checkpoint="latest"

    fi
done
