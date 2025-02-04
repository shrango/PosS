import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()


f = open(args.input_file, 'r')
lines = f.readlines()
print('num of samples:', len(lines))

avg_accept_length = 0

for line in lines:
    data = json.loads(line)
    avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1

avg_accept_length /= len(lines)

print("acceptance length:", avg_accept_length)