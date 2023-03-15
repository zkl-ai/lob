# coding=utf8

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import sys
from data_precess.data import LOBDataset
import logging
from vit import VisionTransformer as ViT
from ocet import OCET

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(0)

torch.cuda.manual_seed(1)

logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


def train(epoch):
    start_time = time.time()
    model.train()
    correct, total = 0.0, 0.0
    loss_epoch = []
    for i, (lob, label) in enumerate(dataloader_train):
        optimizer.zero_grad()
        lob, label = lob.cuda(), label.cuda()
        pred = model(lob)
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()
        pre_cpu = predicted.cpu().numpy()
        loss = loss_fn(pred, label)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logging.info("Epoch: %d/%s, Step: %d/%d,  Loss: %f, Acc: %f", 
                             epoch , epochs, i, len(dataloader_train),
                             float(loss), float(correct / total))
    end_time = time.time()
    train_acc.append(correct / total)
    train_loss.append(np.mean(loss_epoch))
    logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f ",
                     epoch, epochs, train_loss[-1], train_acc[-1])

@torch.no_grad()
def evaling(epoch):
    start_time = time.time()
    model.eval()
    correct = 0.0
    total = 0.0
    loss_epoch = []
    for (lob, label) in tqdm(dataloader_test, ncols=40):
        lob, label = lob.cuda(), label.cuda()
        outputs = model(lob)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += float(torch.sum(predicted == label))

        loss = loss_fn(outputs, label)
        loss_epoch.append(loss.item())


    end_time = time.time()
    curr_acc = correct / total
    logging.info('Evaluating Network.....')
    logging.info('Test set: Epoch: {}, Current Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        curr_acc,
        end_time - start_time
    ))
    test_loss.append(np.mean(loss_epoch))
    curr_loss = test_loss[-1]
    return curr_acc, curr_loss


if __name__ == "__main__":
    '''data'''
    '''
    labels = [10, 20, 30, 50, 100] for k in range(5)
    '''
    k = 3
    dep = 1
    bs = 256
    dataset_train = LOBDataset(k, T=100, split='train')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=bs, shuffle=True)
    dataset_test = LOBDataset(k, T=100, split='test')
    
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=bs, shuffle=False)

    model_name = 'ocet' # ocet deeplob deepfolio 
    save_k = ['k_10', 'k_20', 'k_30', 'k_50', 'k_100']
    save_path = 'model_save/FI2010/FE/100/' + save_k[k] + '/' + model_name + '/'
    os.makedirs(save_path, exist_ok=True)

    '''model'''
    mode = model_name  
# Model parameters = 163666
    model = OCET(
        num_classes=3,
        dim=40,
        depth=2,
        heads=4,
        dim_head=10,
        mlp_dim=80,
    )
    # model = ViT(
    #     in_channels=1,
    #     embedding_dim=32,
    #     num_layers=3,
    #     num_heads=4,
    #     qkv_bias=False,
    #     mlp_ratio=2.0,
    #     dropout_rate=0.1,
    #     num_classes= 3
    # )
    
    model = model.cuda()
  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0001)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    logging.info('  Model = %s', str(model))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Train num = %d', len(dataset_train))
    logging.info('  Test num = %d', len(dataset_test))

    epochs = 100
    train_acc = []
    train_loss = []

    test_acc = []
    best_acc = 0.0
    best_epoch = 1
    best_loss = 2.0
    test_loss = []
    for epoch in range(1, epochs + 1):
        train(epoch)
        curr_acc, curr_loss = evaling(epoch)
        test_acc.append(curr_acc)
        if curr_acc > best_acc:
            best_epoch = epoch
            best_acc = curr_acc
            best_loss = curr_loss
        output_path = save_path + mode + '_epoch_' + str(epoch) + '_valloss_' + str(format(curr_loss, '.4f')) + '_valacc' + '_' + str(format(curr_acc, '.4f')) + '.pth'
        torch.save(model.state_dict(), output_path)
        logging.info("The best epoch is : {}, best Accuracy: {:.4f}, loss is : {:.4f}".format(
            best_epoch,
            best_acc,
            best_loss
        ))
    
