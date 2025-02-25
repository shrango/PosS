CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_manual_lora_ea_answer_llama3chat \
	--ea-model-path /home/jovyan/workspace/project/manual-lora-from-hass/checkpoint/state_0 \
	--base-model-path /home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
	--temperature 0 \
	--model-id manual-lora-from-hass-epoch1 &

CUDA_VISIBLE_DEVICES=1 python -m evaluation.gen_manual_lora_ea_answer_llama3chat \
	--ea-model-path /home/jovyan/workspace/project/manual-lora-from-hass50/checkpoint/state_0 \
	--base-model-path /home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
	--temperature 0 \
	--model-id manual-lora-from-hass50-epoch1 &

CUDA_VISIBLE_DEVICES=2 python -m evaluation.gen_manual_lora_ea_answer_llama3chat \
	--ea-model-path /home/jovyan/workspace/project/manual-lora-from-hass50/checkpoint/state_3 \
	--base-model-path /home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a \
	--temperature 0 \
	--model-id manual-lora-from-hass50-epoch4 &