export PYTHONPATH='/nfs/a100-006/hanweiguang/local/trl/examples'
deepspeed --hostfile accelerate_configs/deepspeed_zero3.yaml ./sentiment_tuning_modify.py