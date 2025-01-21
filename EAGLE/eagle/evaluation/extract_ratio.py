import sys
import json
filename = sys.argv[1]
with open(filename, "r") as f:
    content = f.readlines()

def parse_line(line):
    js = json.loads(line)
    tokens = js["choices"][0]["new_tokens"]
    forward_times = js["choices"][0]["forward_times"]
    if "acc_len_count" in js["choices"][0]:
        acc_len_count = js["choices"][0]["acc_len_count"]
    else:
        acc_len_count = None
    return tokens, forward_times, acc_len_count

tokens = []
times = []
acc_len_counts = {}
for line in content:
    sub_token, sub_time, sub_len_count = parse_line(line)
    tokens.extend(sub_token)
    times.extend(sub_time)
    if sub_len_count is not None:
        for k in sub_len_count:
            acc_len_counts[k] = acc_len_counts.get(k, 0) + sub_len_count[k]
sorted_acc_count = {}
sorted_acc_count_ratio = {}
if len(acc_len_counts):
    for k in sorted(acc_len_counts.keys()):
        sorted_acc_count[k] = acc_len_counts[k]
        sorted_acc_count_ratio[k] = acc_len_counts[k]/sum(acc_len_counts.values())
micro_ratios = []
for i in range(len(tokens)):
    micro_ratios.append(tokens[i]/times[i])

micro_ratio = sum(micro_ratios)/len(micro_ratios)
sum_token = sum(tokens)
sum_time = sum(times)
macro_ratio = sum_token / sum_time

print(f"The micro compression ratio is {micro_ratio}, macro crompression ratio is {macro_ratio}. The accept length count is {sorted_acc_count} and {sorted_acc_count_ratio}")