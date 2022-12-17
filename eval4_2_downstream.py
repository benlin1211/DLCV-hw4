# Ref: https://github.com/lucidrains/byol-pytorch
import torch
import torch.nn as nn
# from byol_pytorch import BYOL
import os
import glob
import random
from PIL import Image
from tqdm import tqdm
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

from BYOL import BYOL # https://arxiv.org/pdf/2006.07733.pdf


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_csv_with_index(path):
    label = []
    filename = []
    with open(f'{path}', newline='') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        # 以迴圈輸出每一列
        for i, row in enumerate(rows):
            if i==0: 
                continue
            # with idx
            idx, f, l = row
            label.append(l)
            filename.append(f)

    return label, filename


class Mini(Dataset):
    def __init__(self, data_path, filenames, transform, label2id):
        super().__init__()
        self.transform = transform
        self.image_paths = [os.path.join(data_path, f) for f in filenames]
        # sorted(glob.glob(os.path.join(data_path, "*.jpg")))

        self.label2id = label2id
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        return img

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


class DownStreamResnet(nn.Module):
    def __init__(self, n_class=65):
        super(DownStreamResnet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)    
        self.nn = nn.Linear(1000, n_class)

    def forward(self, img):
        x = self.resnet(img)
        x = self.nn(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 4-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume_path", help="Checkpoint", default= "./ckpt4_2C/downstring_SGD.pth") 
    parser.add_argument("--csv_path", help="csv_path", default= "./hw4_data/office/val.csv") 
    parser.add_argument("--data_path", help="data_path", default= "./hw4_data/office/val") 
    parser.add_argument("--output_name", help="output_name", default= "./output_p2/test_pred.csv") 
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)

    # ================================= TRAIN =====================================   

    args = parser.parse_args()
    print(vars(args))

    same_seeds(1211)
    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)
    # args
    batch_size = args.batch_size

    resume_path = args.resume_path
    csv_path = args.csv_path
    output_name = args.output_name
    # create folder
    sub = output_name.split("/")[-1]
    output_path = output_name.replace(sub,'')
    os.makedirs(output_path, exist_ok=True)
    data_path = args.data_path

    with open(csv_path) as f:
        myCsv = csv.reader(f)
        filenames = []
        for i, (id, filename, label) in enumerate(myCsv):
            if i==0:
                continue
            else:
                filenames.append(filename)

    # filenames = sorted(filenames)
    print(len(filenames))

    # Transform 
    # it is already in DEFAULT_AUG of BYOL.BYOL .
    # Ref Appendix B. in  https://arxiv.org/pdf/2006.07733.pdf
    tfm = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # create label2id dict
    with open("./p2_label2id.csv") as f:
        myCsv = csv.reader(f)
        label2id = {}
        id2label = {}
        for i, (id, label) in enumerate(myCsv):
            if i==0:
                continue
            else:
                label2id[label] = id
                id2label[int(id)] = label
    # print(label2id['Helmet'])
    # print(id2label[1])
    
    # Dataset
    val_dataset = Mini(data_path, filenames, tfm, label2id)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8)

    print("Val:", len(val_dataloader))
    # model
    model = DownStreamResnet(n_class=65)

    # load pretrain


    print(f"Load from {resume_path}")
    checkpoint = torch.load(resume_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # to device
    # print(model)
    model = model.to(device)
    show_n_param(model)

    result = []
    with torch.no_grad():
        # ========================= Eval ==========================
        model.eval()

        pbar_val = tqdm(val_dataloader)
        loss_val_epoch = []
        for data in pbar_val:
            img = data    
            img = img.to(device)
            # print(label.shape)

            logits = model(img)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            result.append(pred)

    result = np.concatenate(result, axis=0)
    print(result.shape)

    prediction = []
    for i in range(result.shape[0]):
        
        # print(filenames[i])
        _class = id2label[result[i]]
        prediction.append(_class)
        # print(result[i])
        # print(_class)

    df = pd.DataFrame() # apply pd.DataFrame format 
    df["filename"] = filenames
    df["label"] = prediction
    df.to_csv(output_name, index = True)
    
    


        




