import json
from transformers import AutoTokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--baseline_json", type=str)
parser.add_argument("--hass_json", type=str)
args = parser.parse_args()

tokenizer=AutoTokenizer.from_pretrained(args.model_path)
jsonl_file = args.hass_json
jsonl_file_base = args.baseline_json

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


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("speedup ratio:",np.array(speeds).mean()/np.array(speeds0).mean())


