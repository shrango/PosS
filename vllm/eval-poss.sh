for iter in 1
do
    for DATASET in mt_bench alpaca gsm8k qa sum humaneval
    do
        for TEMP in 0 1
        do
            for bs in 1 2 4 8
            do
                CUDA_VISIBLE_DEVICES=0 python run_vllm_eval.py \
                --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
                --dataset $DATASET \
                --experiment-name llama31_poss3_${DATASET}_temp${TEMP}_iter${iter}_bs${bs} \
                --use-eagle3 \
                --eagle3-draft HINT-lab/PosS3-E3-Llama3.1-8B-Instruct \
                --temperature $TEMP \
                --batch-size ${bs}
            done
        done
    done
done