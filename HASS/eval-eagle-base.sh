# pip install --upgrade pip
# pip install setuptools==69.5.1
# pip install -r requirements.txt --use-pep517

## hass

CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_llama3chat \
--ea-model-path /home/jovyan/workspace/project/llama3/standard/checkpoint/state_20 \
--base-model-path /home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
--model-id eagle-baseline \
--bench-name mt_bench \
--temperature 0 