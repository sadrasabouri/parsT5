export model_folder="/home/tensorflow/parsT5/t5-fa-model"
export cache_folder="/home/tensorflow/.cache/t5-fa-model"
export train_file="/home/tensorflow/text-data/informal.txt"

python run_t5_mlm_flax.py \
    --model_name_or_path=$model_folder \
    --output_dir=$model_folder \
    --cache_dir=$cache_folder \
    --model_type="t5" \
    --config_name=$model_folder \
    --tokenizer_name=$model_folder \
    --train_file=$train_file \
    --max_seq_length="256" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --num_train_epochs="10" \
    --eval_steps="10000" \
    --adafactor \
    --max_eval_steps="12000" \
    --learning_rate="0.005" \
    --weight_decay="0.001" \
    --warmup_steps="2000" \
    --logging_steps="500" \
    --save_steps="2500" \
    --resume_from_checkpoint=$model_folder \
    --preprocessing_num_workers=20 \
    # --dataset_name="oscar" \
    # --dataset_config_name="unshuffled_deduplicated_fa" \
