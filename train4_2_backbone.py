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
import matplotlib.pyplot as plt

from BYOL import BYOL # https://arxiv.org/pdf/2006.07733.pdf
# from torchlars import LARS # https://github.com/kakaobrain/torchlars
# from scheduler import cosine

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
    parser.add_argument("--ckpt_path", help="Checkpoint", default= "./ckpt4_2_backbone") 
    
    parser.add_argument("--data_path", help="data_path", default= "./hw4_data/mini/") 
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-2) # 原論文說 0.5*batch_size/256
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=1.5e-6)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=1000) 
    # ================================= TRAIN =====================================   
    parser.add_argument("--stepLR_step", help="learning rate decay factor.",type=int, default=50)
    parser.add_argument("--stepLR_gamma", help="learning rate decay factor.",type=float, default=0.998)
                     
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
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs

    stepLR_step = args.stepLR_step 
    stepLR_gamma = args.stepLR_gamma

    ckpt_path = args.ckpt_path
    os.makedirs(ckpt_path, exist_ok=True)
    data_path = args.data_path

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


    # Dataset
    train_dataset = Mini(data_path, tfm, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    print("Train:", len(train_dataloader))
    # model
    resnet = models.resnet50(pretrained=False)
    learner = BYOL.BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False,       # turn off momentum in the target encoder
    )
    learner = learner.to(device)
    show_n_param(learner)

    # optimizer 
    # Ref: https://github.com/kakaobrain/torchlars
    optimizer = torch.optim.AdamW(learner.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = LARS(optim.SGD(learner.parameters(), lr=lr))

    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepLR_step, stepLR_gamma)
    T_0 = 10 * len(train_dataloader) # The first warmup period
    T_mult = 99 # 10*99 = 990 epoches to go 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    
    
    loss_curve_train = []
    step = 0
    for epoch in range(n_epochs):
        # ========================= Train ==========================
        learner.train()
        print("Train")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch {epoch}|{n_epochs}")
        
        for data in pbar:
            img, label = data    
            img = img.to(device)
            # label = label.to(device)
            loss = learner(img)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # learner.update_moving_average()
            pbar.set_postfix(loss=loss.item(), lr = optimizer.param_groups[0]['lr'])
            loss_curve_train.append(loss.item())
            step+=1


        # Save model
        save_as = os.path.join(ckpt_path, f"BYOL.pth")
        torch.save({
                'epoch': epoch,
                'model_state_dict': learner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_as)
    
    # plot loss 
    x = list(range(0, len(loss_curve_train)))
    plt.plot(x, loss_curve_train)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title("Train 4-2 backbone loss curve")
    plt.savefig("./Train_loss_4_2.png")



