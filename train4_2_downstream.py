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
    def __init__(self, data_path, transform, label2id, mode='train'):
        super().__init__()
        self.transform = transform
        self.data_path = data_path
        self.mode = mode
        # print("data_path", data_path)
        if self.mode == "train" or self.mode == "val": 
            path = os.path.join(self.data_path, f"{self.mode}.csv")
            self.labels, image_names = read_csv_with_index(path)
            self.image_paths = [os.path.join(data_path, self.mode, n) for n in image_names]
        else: # test
            self.image_paths = glob.glob(os.path.join(self.data_path, "*.jpg"))
        self.label2id = label2id
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        if self.mode == "train" or self.mode == "val": 
            label = self.labels[idx]

            return img, int(label2id[label])
        else: # test 
            return img

def show_n_param(model):
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")


class DownStreamResnet(nn.Module):
    def __init__(self, n_class=65, is_pretrain=False):
        super(DownStreamResnet, self).__init__()
        self.resnet = models.resnet50(pretrained=is_pretrain)    
        self.nn = nn.Linear(1000, n_class)

    def forward(self, img):
        x = self.resnet(img)
        x = self.nn(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hw 4-2 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ================================= Report setting =====================================   
    parser.add_argument("--freeze_backbone", help="freeze_backbone", default=False)
    parser.add_argument("--load_pretrain", help="load_pretrain", default=True)
    
    parser.add_argument("--pretrain_path", help="Checkpoint", default= "./ckpt4_2_naive_backbone/improved-net.pt") 
    parser.add_argument("--ckpt_path", help="Checkpoint", default= "./ckpt4_2_downstream") 
    
    # ================================= TRAIN ===================================== 
    parser.add_argument("--data_path", help="data_path", default= "./hw4_data/office/") 
    parser.add_argument("--batch_size", help="batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=3e-4)
    # parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-9)
    parser.add_argument("--n_epochs", help="n_epochs", type=int, default=100) 
      
    parser.add_argument("--stepLR_step", help="learning rate decay factor.",type=int, default=100)
    parser.add_argument("--stepLR_gamma", help="learning rate decay factor.",type=float, default=0.99)
                     
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
    # device = torch.device("cuda")
    print("Using", device)
    # args
    batch_size = args.batch_size
    lr = args.learning_rate
    # weight_decay = args.weight_decay
    n_epochs = args.n_epochs

    stepLR_step = args.stepLR_step 
    stepLR_gamma = args.stepLR_gamma

    load_pretrain = args.load_pretrain
    pretrain_path = args.pretrain_path
    freeze_backbone = args.freeze_backbone
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

    # create label2id dict
    with open("./p2_label2id.csv") as f:
        myCsv = csv.reader(f)
        label2id = {}
        for i, (id, label) in enumerate(myCsv):
            if i==0:
                continue
            else:
                label2id[label] = id
    print(label2id['Helmet'])
    
    # Dataset
    train_dataset = Mini(data_path, tfm, label2id, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
    val_dataset = Mini(data_path, tfm, label2id, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8)
    print("Train:", len(train_dataloader))
    print("Val:", len(val_dataloader))
    # model
    model = DownStreamResnet(n_class=65, is_pretrain=False)

    # load pretrain
    if load_pretrain == True:
        print(f"Load from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location = device)
        model.resnet.load_state_dict(checkpoint)
    else: 
        print("No pretrain resnet.")
    
    if freeze_backbone == True:
        print(f"Freeze Backbone")
        for param in model.resnet.parameters():
            param.requires_grad = False
    else:
        print("No freeze backbone.")  

    # to device
    # print(model)
    model = model.to(device)
    show_n_param(model)

    # optimizer 
    # Ref: https://github.com/kakaobrain/torchlars
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr , momentum = 0.9)
    # optimizer = LARS(optim.SGD(learner.parameters(), lr=lr))

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepLR_step, stepLR_gamma)
    # T_0 = 1 * len(train_dataloader) # The first warmup period
    # T_mult = 2
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
    

    
    loss_curve_train = []
    loss_curve_val = []
    loss_best = 100
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        # ========================= Train ==========================
        model.train()
        print("Train")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch {epoch}|{n_epochs}")
        loss_epoch_train = []
        for data in pbar:
            img, label = data    
            img = img.to(device)
            # print(label.shape)

            label = label.to(device)
            logits = model(img)
            # pred = torch.argmax(logits, dim=1)
            # print(pred)

            # print(pred.shape)
            loss = criterion(logits, label)

        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # learner.update_moving_average()
            loss_epoch_train.append(loss.item())
            pbar.set_postfix(loss=loss.item(), lr = optimizer.param_groups[0]['lr'])

        loss_train = sum(loss_epoch_train)/len(loss_epoch_train)
        loss_curve_train.append(loss_train)
        print(f"Epoch {epoch} Train loss: {loss_train:.3f}")
        # ========================= Eval ==========================
        model.eval()
        print("Eval")
        pbar_val = tqdm(val_dataloader)
        pbar_val.set_description(f"Epoch {epoch}|{n_epochs}")
        loss_epoch_val = []
        count=0
        total=0
        for data in pbar_val:
            img, label = data    
            img = img.to(device)
            # print(label.shape)

            label = label.to(device)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

            for p,l in zip(pred, label):
                if p==l:
                    count+=1
                total+=1
                
            loss = criterion(logits, label)

            # learner.update_moving_average()
            pbar.set_postfix(loss=loss.item(), lr = optimizer.param_groups[0]['lr'])
            loss_epoch_val.append(loss.item())

        # Save model
        loss_val = sum(loss_epoch_val)/len(loss_epoch_val)
        loss_curve_val.append(loss_val)
        
        print(f"Epoch {epoch} Eval loss: {loss_val:.3f}, Acc: {count/total}")
        if loss_best > loss_val:
            print("Save model")
            loss_best = loss_val
            save_as = os.path.join(ckpt_path, f"downstring_SGD.pth")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, save_as)
    
    # # plot loss --n_epochs
    x = list(range(0, n_epochs))
    plt.plot(x, loss_curve_train, label='Train loss', marker = 'o')
    plt.plot(x, loss_curve_val, label='Valid loss', marker = 'o')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Train 4-2 backbone loss curve")
    save_img_as = os.path.join(ckpt_path,"loss.png")
    plt.savefig(save_img_as)

    print(f"Loss curve is saved at {save_img_as}")



