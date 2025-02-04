import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()


f = open(args.input_file, 'r')
lines = f.readlines()
print('num of samples:', len(lines))

avg_accept_length = 0

def groupby_accept_length(data):
    res = {}
    for d in data:
        if d['accept_length'] not in res:
            res[d['accept_length']] = 0
        res[d['accept_length']] += 1
    return res

group_all = {}
for line in lines:
    data = json.loads(line)
    subgroup = groupby_accept_length(data['choices'])
    for k, v in subgroup.items():
        if k not in group_all:
            group_all[k] = 0
        group_all[k] += v
    avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1
    compare_avg = sum([i*subgroup[i] for i in subgroup]) / sum(subgroup.values())
    assert abs(avg_accept_length - compare_avg) < 1e-5

avg_accept_length /= len(lines)

print(f"acceptance length:{avg_accept_length}, group_all: {group_all}")