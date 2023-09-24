# Configuration here
export ID="5" # -- Identity that will be evaluated on
export ROOT="outputs/DB" # -- Root folder that contains all folders to be evaluated on
export IMAGE_DIR="checkpoint-1000/images" # -- Exact path to the image_dirs inside the identity folder

if [ "$ID" = "all" ]; then
    for dir in "$ROOT"/*
    do
        if [ -d "$dir" ]; then
            dir_name=$(basename $dir)
            reference_dir="db_dataset/$dir_name/set_A"
            image_dir="$ROOT/$dir_name/$IMAGE_DIR"

            for prompt_dir in "$image_dir"/*
            do
                prompt_dir_name=$(basename $prompt_dir)
                target_dir="$image_dir/$prompt_dir_name/all"

                echo "Eval on set $dir_name, prompt '$prompt_dir_name'"

                python src/eval.py \
                    --target_dir="$target_dir" \
                    --reference_dir="$reference_dir"
            done
        fi
    done
else
    reference_dir="db_dataset/$ID/set_A"
    image_dir="$ROOT/$ID/$IMAGE_DIR"

    for dir in "$image_dir"/*
    do
        dir_name=$(basename $dir)

        target_dir="$image_dir/$dir_name/all"

        echo "Eval on set $ID"

        python src/eval.py \
            --target_dir="$target_dir" \
            --reference_dir="$reference_dir"
    done
fi