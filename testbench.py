import json
import os

test_json = os.path.join("./hw4_data/hotdog/transforms_val.json")
with open(test_json) as f:
    data = json.load(f)

filename = []
for frame in data["frames"]:
    file_path = frame['file_path']
    filename.append(file_path.split("/")[-1])

print(filename)
print(len(filename))