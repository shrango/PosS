#pip install --upgrade pip
#pip install setuptools==69.5.1
#pip install -r requirements.txt --use-pep517

ckpt=/storage1/jiaxinh/Active/langlin/project/agent2-tiny-from-scratch/checkpoint
mkdir -p $ckpt
prep=/storage1/jiaxinh/Active/langlin/Dataset/ge_data/sharegpt_0_67999_mufp16
#prep=/home/jovyan/workspace/preprocess/mini_trainset
BasePath=/storage1/jiaxinh/Active/models_cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a
# restore_from=/home/jovyan/workspace/project/llama3/standard/checkpoint/state_20
#     --ckpt_path $restore_from \

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29501 -m --mixed_precision=bf16 train.main_agent_layer \
    --basepath $BasePath \
    --tmpdir $prep \
    --cpdir $ckpt \
    --configpath train/EAGLE-LLaMA3-Instruct-8B \
    --epoch 20 \
    --bs 4 \
    --topk 10 \
    --topk_w 0 \
    --lr 3e-5 \
    --forward_num_total 10 \
    --position_per_layer 2 \
    --draft_model_size "small"
