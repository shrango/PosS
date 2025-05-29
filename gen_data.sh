python -m ge_data.allocation \
    --base_model "llama3" \
    --model_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "ge_data/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
    --outdir training-data-llama3

# python -m ge_data.allocation \
#     --base_model "llama2" \
#     --model_path "meta-llama/Llama-2-13b-chat-hf" \
#     --data_path "ge_data/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
#     --outdir training-data-llama2