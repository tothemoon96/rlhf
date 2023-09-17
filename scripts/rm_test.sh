export CUDA_VISIBLE_DEVICES="0,2,3,4"
gpus=4

PROJECT_DIR="/workdir/rlhf"
export PYTHONPATH="$PROJECT_DIR/rlhf"

export WANDB_PROJECT=test_rm
export WANDB_RUN_ID=test
export WANDB_RESUME=allow

model_name_or_path="bigscience/bloomz-560m"
output_dir="$PROJECT_DIR/saved_models/$WANDB_PROJECT/$WANDB_RUN_ID"
mkdir -p ${output_dir}

train_file=$PROJECT_DIR/data/test_data/test_rm.jsonl
validation_file=$PROJECT_DIR/data/test_data/test_rm.jsonl
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=64

# accelerate launch \
#     --config_file configs/accelerate_config.yaml \
#     --num_processes $gpus \
#     "rm/train_rm.py" \
#     --model_name $model_name_or_path \
#     --train_data $train_file \
#     --eval_data $validation_file \
#     --cache_dir $cache_dir \
#     --report_to "tensorboard" \
#     --logging_steps 1 \
#     --learning_rate 1e-5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --num_train_epochs 1 \
#     --seq_length $cutoff_len \
#     --gradient_accumulation_steps 1 \
#     --gradient_checkpointing False \
#     --load_in_8bit False \
#     --load_in_4bit False \
#     --use_peft False \
#     --trust_remote_code True \
#     --output_dir $output_dir \
#     --use_llama False

torchrun --nproc_per_node $gpus --master-port 29600 \
    "rm/train_rm_bug.py" \
    --deepspeed "configs/deepspeed_config.json" \
    --model_name $model_name_or_path \
    --train_data $train_file \
    --eval_data $validation_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --max_length $cutoff_len \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --load_in_8bit False \
    --load_in_4bit False \
    --use_peft False \
    --trust_remote_code True \
    --output_dir $output_dir \
    --use_llama False