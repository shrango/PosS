from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdb

model = EaModel.from_pretrained(
    base_model_path="/home/jovyan/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a",
    ea_model_path="/home/jovyan/workspace/ckpt",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

bigname = "meta-llama/Meta-Llama-3-8B-Instruct"
bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",torch_dtype=torch.float16)


your_message="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

pdb.set_trace()
outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
outs_ea = model(input_ids.cuda(), output_orig=True)

hs_big = outs_big.hidden_states[-1]
hs_ea = outs_ea[-1]

