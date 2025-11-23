#!/bin/bash

set -x
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/user/zhangchi/mathllm/Perception_rl/PEARL
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/user/zhangchi/conda_init_rc_ana_h
conda activate PEARL
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/user/zhangchi/Init-models/hfmodels/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
nnodes=1
TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=256
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
n=5

CONGI_FILE="examples/configs/config_pearl.yaml"
TRAIN_FILE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/user/zhangchi/mathllm/Perception_rl/PAPO/data/PAPO_ViRL39K_train_probe_v2_multiqa/train.parquet"
VAL_FILE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/user/zhangchi/mathllm/Perception_rl/PAPO/data/PAPO_MMK12_test/test.parquet"
probe_weight=0.95
enable_probe_loss=true
probe_loss_coef=0.1
enable_stage1=false
enable_trim=true
enable_reweight=true
mixfilter=true
lr_scheduler_type=constant
if [ "$mixfilter" = true ]; then
    filter_key=mixfilter
else
    filter_key=og_accuracy
fi
if [ "$enable_stage1" = true ]; then
    echo "enable_stage1 is true, probe_loss_coef is ignored"
else
    echo "enable_stage1 is false, probe_loss_coef is used"
fi
echo "probe_weight is useless in current version"
EXP_NAME="pr_v2mqa_onestage_virl39k_7b_probecoed${probe_loss_coef}_mixfilter${mixfilter}_reweight${enable_reweight}_trim${enable_trim}_lr${lr_scheduler_type}"
FORMAT_PROMPT="examples/format_prompt/math.jinja"
REWARD_FUNCTION="examples/reward_function/math_mqa.py:compute_score"
MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=2
MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=8
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    worker.rollout.n=${n} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE} \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=${nnodes} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.val_freq=10 \
    trainer.save_freq=5 \
    trainer.save_limit=30 \
    worker.actor.enable_stage1=${enable_stage1} \
    algorithm.filter_key=${filter_key} \
    worker.reward.reward_function_kwargs.probe_weight=${probe_weight} \
    algorithm.enable_probe_loss=${enable_probe_loss} \
    worker.actor.probe_loss_coef=${probe_loss_coef} \
    algorithm.enable_reweight=${enable_reweight} \
    algorithm.enable_trim=${enable_trim} \
    worker.actor.optim.lr_scheduler_type=${lr_scheduler_type}
