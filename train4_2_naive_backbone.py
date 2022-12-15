import torch
import torch.nn as nn
from BYOL import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import glob
import random
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


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
    def __init__(self, data_path, transform, mode='train'):
        super().__init__()
        self.transform = transform
        self.data_path = data_path
        self.mode = mode
        # print("data_path", data_path)
        if self.mode == "train": 
            path = os.path.join(self.data_path, "train.csv")
            self.labels, image_names = read_csv_with_index(path)
            self.image_paths = [os.path.join(data_path, "train", n) for n in image_names]
        else: # test
            self.image_paths = glob.glob(os.path.join(self.data_path, "*.jpg"))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        if self.mode == "train" or self.mode == "val": 
            label = self.labels[idx]
            return img, label
        else: # test 
            return img

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 4-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt_path", help="Checkpoint", default= "./ckpt4_2_backbone_005") 
    
    parser.add_argument("--data_path", help="data_path", default= "./hw4_data/mini/") 
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.05)
    # parser.add_argument("--weight_decay", help="weight decay", type=float, default=1.5e-6)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=100) 

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
    lr = args.learning_rate
    # weight_decay = args.weight_decay
    n_epochs = args.n_epochs

    ckpt_path = args.ckpt_path
    os.makedirs(ckpt_path, exist_ok=True)
    data_path = args.data_path


    tfm = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])


    # Dataset
    train_dataset = Mini(data_path, tfm, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    print("Train:", len(train_dataloader))
    # model
    resnet = models.resnet50(pretrained=True)
    learner = BYOL.BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )
    learner = learner.to(device)
    show_n_param(learner)

    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    loss_curve_train = []
    step = 0
    for epoch in range(n_epochs):
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch {epoch}|{n_epochs}")
        
        for data in pbar:
            img, _ = data # we don't need label!
            img = img.to(device)
            loss = learner(img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

    # save your improved network
    save_as = os.path.join(ckpt_path,'improved-net.pt')
    print(f"Save at {save_as}")
    torch.save(resnet.state_dict(), save_as)