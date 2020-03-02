import torch
import os
import numpy as np
from config import CFG, logger
from Net_test import do_test

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result = [] # Save the results

def do_train(model, optimizer, dataloader_src, dataloader_tar):
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    for epoch in range(1, CFG['epoch'] + 1):
        model.train()
        len_dataloader = min(len(dataloader_src), len(dataloader_tar))
        data_src_iter = iter(dataloader_src)
        data_tar_iter = iter(dataloader_tar)

        i = 1
        while i < len_dataloader + 1:
            p = float(i + epoch * len_dataloader) / CFG['epoch'] / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Training model using source data
            data_source = data_src_iter.next()
            optimizer.zero_grad()
            s_img, s_label = data_source[0].to(DEVICE), data_source[1].to(DEVICE)
            domain_label = torch.zeros(CFG['batch_size']).long().to(DEVICE)
            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img = data_target[0].to(DEVICE)
            domain_label = torch.ones(CFG['batch_size']).long().to(DEVICE)
            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_t_domain + err_s_domain

            err.backward()
            optimizer.step()

            if i % CFG['log_interval'] == 0:
                logger.info(
                    'Epoch: [{}/{}], Batch: [{}/{}], err_s_label: {:.4f}, err_s_domain: {:.4f}, err_t_domain: {:.4f}'.format(
                        epoch, CFG['epoch'], i, len_dataloader, err_s_label.item(), err_s_domain.item(),
                        err_t_domain.item()))
            i += 1

        # Save model, and test using the source and target
        if not os.path.exists(CFG['checkponit']):
            os.mkdir(CFG['checkponit'])
        torch.save(model, '{}/mnist_mnistm_model_epoch_{}.pth'.format(CFG['checkponit'], epoch))
        acc_src = do_test(model, CFG['src_img_path'], epoch)
        acc_tar = do_test(model, CFG['tar_img_path'], epoch)
        result.append([acc_src, acc_tar])
        np.savetxt('result.csv', np.array(result), fmt='%.4f', delimiter=',')