import json
import os

test_json = os.path.join("./hw4_data/hotdog/transforms_val.json")
with open(test_json) as f:
    data = json.load(f)

filename = []
for frame in data["frames"]:
    filename.append(frame['file_path'])

print(filename)
print(len(filename))