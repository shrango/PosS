# This is a training demostration for Llama2-13B with PosS-3.
# To train models with Llama3-8B, please change the llama2 related things to llama3.

# "position_per_layer" means the number of positions assigned to one layer. i.e. "position_per_layer N" stands for method "PosS-N".
# "forward_num_total" means the training depth. The number of layers in the model is "forward_num_total / position_per_layer".
saveckpt=checkpoints/llama2-13b-chat/poss-3
mkdir -p $saveckpt

# Please follow EAGLE to generate training data, and fill the path below
prep=/PATH/to/training-data/llama2-13b/sharegpt_0_67999_mufp16
BasePath=/PATH/to/your/model-cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8
# Continue training from EAGLE model to shorten the training time.
restore_from=/PATH/to/eagle/e.g./yuhuili/EAGLE-llama2-chat-13B

# Layers = forward_num_total / position_per_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m --mixed_precision=bf16 train.main_poss \
    --basepath $BasePath \
    --tmpdir $prep \
    --cpdir $saveckpt \
    --ckpt_path $restore_from \
    --configpath train/llama_2_chat_13B_config.json \
    --epoch 20 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --lr 3e-5 \
    --forward_num_total 6 \
    --position_per_layer 3
