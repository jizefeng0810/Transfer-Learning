import torch
import os
import numpy as np
from config import CFG, logger
from torchvision import datasets
from torchvision import transforms
from pre_data.data_load import GetLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_data(dataset_name):
    img_transform = transforms.Compose([
        transforms.Resize(CFG['batch_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if dataset_name == 'mnist_m':

        dataset = GetLoader(
            data_root=CFG['test_img_path'],
            data_list=CFG['test_list'],
            transform=img_transform
        )
    else:
        dataset = datasets.MNIST(
            root=CFG['src_img_path'],
            train=False,
            transform=img_transform,
            download=True
        )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=8
    )
    return dataloader

def do_test(model, dataset_name, epoch):
    alpha = 0
    dataloader = load_test_data(dataset_name)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output, _ = model(input_data=t_img, alpha=alpha)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / len(dataloader.dataset) * 100
    logger.info('Epoch: [{}/{}], accuracy on {} dataset: {:.4f}%'.format(epoch, CFG['epoch'], dataset_name, accu))
    return accu