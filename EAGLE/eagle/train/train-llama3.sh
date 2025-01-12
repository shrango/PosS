ckpt=/home/jovyan/workspace/project/llama3/standard/checkpoint
mkdir -p $ckpt
prep=/home/jovyan/workspace/preprocess/sharegpt_0_67999_mufp16

accelerate launch -m --mixed_precision=bf16 eagle.train.main --tmpdir $prep \
--cpdir $ckpt --configpath EAGLE-LLaMA3-Instruct-8B