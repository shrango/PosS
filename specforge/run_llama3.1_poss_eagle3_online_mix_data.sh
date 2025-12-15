# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# ROOT_DIR=$(dirname $SCRIPT_DIR)
# export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# # train eagle3 for llama3.1-8b
# NUM_GPUS=${1:-8}
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_poss_eagle3_online.py \
    --target-model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --train-data-path /Path-to/mix_sharegpt_ultrachat.jsonl \
    --output-dir /Path-to/checkpoints/llama31-poss-eagle3 \
    --num-epochs 20 \
    --batch-size 4 \
    --warmup-ratio 0.015 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --dist-timeout 240 \
    --chat-template llama3 \
    --cache-dir /Path-to/models_cache \
    --attention-backend flex_attention \
    --ttt-length 8 \
    --loss-weight-decay 0.9 \
    --resume \
    --wandb \
    --wandb-key Your wandb key \
    --wandb-project poss-eagle3 \
    --vocab_mapping_path /Path-to/checkpoints/llama31-poss-eagle3/vocab_mapping_llama31.pt

# If resume training from EAGLE-3 pretrained checkpoint, please save its "t2d" and "d2t" mapping into vocab_mapping_llama31.pt
# otherwise, please remove "--resume" and "--vocab_mapping_path"
