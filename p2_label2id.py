import csv
import pandas as pd

with open("./hw4_data/office/train.csv") as f:
    myCsv = csv.reader(f)


    label = []
    for i, row in enumerate(myCsv):
        if i==0:
            continue
        if row[2] not in label:
            label.append(row[2])
    
    print(label)
    print(len(label))
        

    df = pd.DataFrame(label)
    df.to_csv('p2_label2id.csv', index=True)

with open("./p2_label2id.csv") as f:
    myCsv = csv.reader(f)
    myDict = {}
    for i, (id, label) in enumerate(myCsv):
        if i==0:
            continue
        else:
            myDict[label] = id
    

print(myDict)
print(myDict["Helmet"])
    
