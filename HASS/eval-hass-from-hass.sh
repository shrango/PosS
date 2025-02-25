# pip install --upgrade pip
# pip install setuptools==69.5.1
# pip install -r requirements.txt --use-pep517

## hass

CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_llama3chat \
--ea-model-path /home/jovyan/workspace/project/hass-from-hass/checkpoint/state_10 \
--base-model-path /home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
--model-id llama3-hass-from-hass \
--bench-name mt_bench \
--temperature 0 \

# CUDA_VISIBLE_DEVICES=1 python -m evaluation.gen_ea_answer_llama2chat \
# --ea-model-path \
# --base-model-path \
# --model-id llama2-chat-7b_hass \
# --bench-name mt_bench \
# --temperature 1 \
# --seed 42

# ## baseline

# CUDA_VISIBLE_DEVICES=2 python -m evaluation.gen_baseline_answer_llama2chat \
# --ea-model-path \
# --base-model-path \
# --model-id llama2-chat-7b_naive\
# --bench-name mt_bench \
# --temperature 0 \
# --seed 42

# CUDA_VISIBLE_DEVICES=3 python -m evaluation.gen_baseline_answer_llama2chat \
# --ea-model-path \
# --base-model-path \
# --model-id llama2-chat-7b_naive \
# --bench-name mt_bench \
# --temperature 1 \
# --seed 42

# ## acceptance length

# python -m evaluation.acceptance_length \
# --input_file mt_bench/llama2-chat-7b_hass-temperature-0.0.jsonl

# python -m evaluation.acceptance_length \
# --input_file mt_bench/llama2-chat-7b_hass-temperature-1.0.jsonl

# ## speedup ratio

# python -m evaluation.speed \
# --model_path \
# --baseline_json mt_bench/llama2-chat-7b_naive-temperature-0.0.jsonl \
# --hass_json mt_bench/llama2-chat-7b_hass-temperature-0.0.jsonl

# python -m evaluation.speed \
# --model_path \
# --baseline_json mt_bench/llama2-chat-7b_naive-temperature-1.0.jsonl \
# --hass_json mt_bench/llama2-chat-7b_hass-temperature-1.0.jsonl