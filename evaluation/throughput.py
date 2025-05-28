import json
from transformers import AutoTokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--my_model_json", type=str)
args = parser.parse_args()

tokenizer=AutoTokenizer.from_pretrained(args.model_path)
jsonl_file = args.my_model_json

data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)

print('Throughput:',np.array(speeds).mean(), "tokens/s")


