pip install --upgrade pip
pip install setuptools==69.5.1
pip install -r requirements.txt --use-pep517

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_hass \
    --basepath \
    --tmpdir \
    --cpdir \
    --configpath \
    --epoch 40 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch -m --mixed_precision=bf16 train.main_hass \
    --basepath \
    --tmpdir \
    --cpdir \
    --configpath \
    --epoch 10 \
    --lr 0.00001 \
    --bs 2 \
    --topk 10 \
    --topk_w 1 \
    --forward_num_total 3 \
    --ckpt_path 
