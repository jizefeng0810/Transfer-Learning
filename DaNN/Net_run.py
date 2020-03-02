import os
import torch
import torchvision
from model.models import DANN
from Net_train import do_train
from config import CFG, logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from pre_data.data_load import GetLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    img_transform = transforms.Compose([
        transforms.Resize(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    dataset_source = datasets.MNIST(
        root=CFG['src_img_path'],
        train=True,
        transform=img_transform,
        download=True
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=CFG['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4)
    train_list = CFG['tar_img_path'] + '/mnist_m_train_labels.txt'
    dataset_target = GetLoader(
        data_root=CFG['tar_img_path'] + '/mnist_m_train',
        data_list=train_list,
        transform=img_transform
    )
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=CFG['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=8)
    return dataloader_source, dataloader_target


if __name__ == '__main__':
    torch.random.manual_seed(100)
    loader_src, loader_tar = load_data()
    model = DANN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    do_train(model, optimizer, loader_src, loader_tar)