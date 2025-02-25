#pip install --upgrade pip
#pip install setuptools==69.5.1
#pip install -r requirements.txt --use-pep517

ckpt=/home/jovyan/workspace/project/full-from-eagle/checkpoint
mkdir -p $ckpt
prep=/home/jovyan/workspace/preprocess/sharegpt_0_67999_mufp16
#prep=/home/jovyan/workspace/preprocess/mini_trainset
BasePath=/home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a
restore_from=/home/jovyan/workspace/project/llama3/standard/checkpoint/state_20

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29501 -m --mixed_precision=bf16 train.main_full \
    --basepath $BasePath \
    --tmpdir $prep \
    --cpdir $ckpt \
    --ckpt_path $restore_from \
    --configpath train/EAGLE-LLaMA3-Instruct-8B \
    --epoch 20 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --lr 3e-5 \
    --forward_num_total 5
