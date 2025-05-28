import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my_model_json", type=str, help="PATH/to/PosS/mt_bench(or other bench)/my_model.jsonl")
args = parser.parse_args()


f = open(args.my_model_json, 'r')
lines = f.readlines()
print('num of samples:', len(lines))

avg_accept_length = 0

def groupby_accept_length(data):
    res = {}
    for d in data:
        if d not in res:
            res[d] = 0
        res[d] += 1
    return res

group_all = {}
sum_accept_length = 0
for line in lines:
    data = json.loads(line)
    subgroup = groupby_accept_length(data['choices'][0]['accept_length'])
    for k, v in subgroup.items():
        if k not in group_all:
            group_all[k] = 0
        group_all[k] += v
    print(subgroup)
    avg_accept_length = sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1
    sum_accept_length += avg_accept_length
    compare_avg = sum([i*subgroup[i] for i in subgroup]) / sum(subgroup.values()) + 1
    # print(f"avg_accept_length: {avg_accept_length}, compare_avg: {compare_avg}")
    assert abs(avg_accept_length - compare_avg) < 1e-5

avg_accept_length = sum_accept_length/len(lines)

print(f"acceptance length:{avg_accept_length}, group_all: {group_all}")