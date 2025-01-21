import sys
import json
filename = sys.argv[1]
with open(filename, "r") as f:
    content = f.readlines()

def parse_line(line):
    js = json.loads(line)
    tokens = js["choices"][0]["new_tokens"]
    forward_times = js["choices"][0]["forward_times"]
    return tokens, forward_times

tokens = []
times = []
for line in content:
    sub_token, sub_time = parse_line(line)
    tokens.extend(sub_token)
    times.extend(sub_time)

micro_ratios = []
for i in range(len(tokens)):
    micro_ratios.append(tokens[i]/times[i])

micro_ratio = sum(micro_ratios)/len(micro_ratios)
sum_token = sum(tokens)
sum_time = sum(times)
macro_ratio = sum_token / sum_time

print(f"The micro compression ratio is {micro_ratio}, macro crompression ratio is {macro_ratio}.")