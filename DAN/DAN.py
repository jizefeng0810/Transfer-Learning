import torch
import os
import math
import data_loader
import models
from config import CFG
import mmd
from tensorboardX import SummaryWriter



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model, target_test_loader):
    model.eval()
    test_loss = mmd.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    return  correct,len_target_dataset


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    train_loss_clf = mmd.AverageMeter()
    train_loss_transfer = mmd.AverageMeter()
    train_loss_total = mmd.AverageMeter()
    correct = 0
    for e in range(CFG['epoch']):
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            # label_source_pred = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            # loss = clf_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            # writer.add_scalar('data/train_loss_total', train_loss_total.avg, i + e * n_batch)
            # writer.add_scalar('data/train_loss_clf', train_loss_clf.avg, i + e * n_batch)
            # writer.add_scalar('data/train_loss_transfer', train_loss_transfer.avg, i + e * n_batch)
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1, CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
                # print(
                #     'Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, total_Loss: {:.6f}'.format(
                #         e + 1, CFG['epoch'],
                #         int(100. * i / n_batch), train_loss_clf.avg, train_loss_total.avg))
        # Test
        test_correct,len_target_dataset = test(model, target_test_loader)
        print('{} --> {}: current correct: {}, accuracy{: .2f}%\n'.format(
            source_name, target_name, test_correct, 100. * test_correct / len_target_dataset))
        if test_correct > correct:
            correct = test_correct
        print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_target_dataset))
    # writer.close()

def load_data(src, tar, root_dir):
    folder_src = root_dir + src + '/images/'
    folder_tar = root_dir + tar + '/images/'
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.conv1.parameters(),'lr':0},   # frozon
        {'params': model.base_network.bn1.parameters(), 'lr': 0},
        {'params': model.base_network.relu.parameters(), 'lr': 0},
        {'params': model.base_network.maxpool.parameters(), 'lr': 0},
        {'params': model.base_network.layer1.parameters()},         # finetune
        {'params': model.base_network.layer2.parameters()},
        {'params': model.base_network.layer3.parameters()},
        {'params': model.base_network.layer4.parameters()},
        {'params': model.base_network.avgpool.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    print(model)

    # dslr\amazon\webcam
    domains = ['dslr', 'amazon', 'webcam']
    # for i in range(len(domains)):
    #     for j in range(len(domains)):
    #         if i != j:
    #             source_name = domains[i]
    #             target_name = domains[j]
    #             print('Src: %s, Tar: %s' % (source_name, target_name))
    #             log_dir = './data_logs/logs_noDAN_' + domains[i] + '_' + domains[j]
    #             writer = SummaryWriter(log_dir=log_dir)
    #
    #             source_loader, target_train_loader, target_test_loader = load_data(
    #                 source_name, target_name, CFG['data_path'])
    #             train(source_loader, target_train_loader,
    #                   target_test_loader, model, optimizer, CFG, writer)
    source_name = 'webcam'
    target_name = 'amazon'
    print('Src: %s, Tar: %s' % (source_name, target_name))
    # log_dir = './data_logs/logs_DAN1_' + source_name + '_' + target_name
    # writer = SummaryWriter(log_dir=log_dir)

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, CFG['data_path'])
    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)