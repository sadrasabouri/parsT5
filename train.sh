MODEL_DIR="/home/tensorflow/parsT5/t5-fa"
CACHE_DIR="/home/tensorflow/.cache/t5-fa/puramini"
TRAINING_FILE="/mnt/t5-fa/data/all_train.txt"
VALIDATION_FILE="/mnt/t5-fa/data/all_test.txt"

python run_t5_mlm_flax.py \
    --model_name_or_path=$MODEL_DIR \
    --output_dir=$MODEL_DIR \
    --cache_dir=$CACHE_DIR \
    --model_type="t5" \
    --config_name=$MODEL_DIR \
    --tokenizer_name=$MODEL_DIR \
    --train_file=$TRAINING_FILE \
    --validation_file=$VALIDATION_FILE \
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
    --resume_from_checkpoint=$MODEL_DIR \
    --preprocessing_num_workers=20 \
    # --dataset_name="oscar" \
    # --dataset_config_name="unshuffled_deduplicated_fa" \
