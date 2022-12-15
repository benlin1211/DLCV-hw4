import csv
import re


with open('./output_p2/test_pred.csv') as file:
    csvreader = csv.reader(file)
    total = 0
    count = 0
    for row in csvreader:
        id, filename, pred = row
        # https://stackoverflow.com/questions/8270784/how-to-split-a-string-between-letters-and-digits-or-between-digits-and-letters
        print(filename)
        gth = re.split('(\d+)', filename)[0]
        print(gth, pred)
        if gth == pred:
            count += 1
        total+=1
    print(f"result: {count/total} ({count}|{total})")