import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

from vit import VisionTransformer as ViT
from ocet import OCET

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N] - 1

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k]
        self.length = len(x)

#         self.x = torch.from_numpy(x)
#         self.y = torch.from_numpy(y)
        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


def data_preparation():
#     root_train = '/tf/data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training'
#     root_test = '/tf/data/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing'
#     train_data_path = root_train + '/Train_Dst_NoAuction_ZScore_CF_7.txt'
#     test_data_path1 = root_test + '/Test_Dst_NoAuction_ZScore_CF_7.txt'
#     test_data_path2 = root_test + '/Test_Dst_NoAuction_ZScore_CF_8.txt'
#     test_data_path3 = root_test + '/Test_Dst_NoAuction_ZScore_CF_9.txt'
    root_train = '/tf/data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training'
    root_test = '/tf/data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Testing'
    train_data_path = root_train + '/Train_Dst_NoAuction_DecPre_CF_7.txt'
    test_data_path1 = root_test + '/Test_Dst_NoAuction_DecPre_CF_7.txt'
    test_data_path2 = root_test + '/Test_Dst_NoAuction_DecPre_CF_8.txt'
    test_data_path3 = root_test + '/Test_Dst_NoAuction_DecPre_CF_9.txt'
    dec_data = np.loadtxt(train_data_path)
    dec_test1 = np.loadtxt(test_data_path1)
    dec_test2 = np.loadtxt(test_data_path2)
    dec_test3 = np.loadtxt(test_data_path3)
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    batch_size = 32

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    print(dataset_train.x.shape, dataset_train.y.shape)
    tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    return (train_loader, val_loader, test_loader)

# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
        
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')
        
        dt = datetime.now() - t0
        torch.save(model, './cur_iter_model_pytorch')
        np.savetxt('train_losses_vit0.01_10.txt', train_losses)
        np.savetxt('test_losses_vit0.01_k10.txt', test_losses)
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses

def compute_acc(test_loader):
    model = torch.load('best_val_model_pytorch')

    n_correct = 0.
    n_total = 0.
    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = F.softmax(model(inputs), dim=1)
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total
    print(f"Test acc: {test_acc:.4f}")

def compute_metric(test_loader):
    model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = F.softmax(model(inputs), dim=1)
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)    
    all_predictions = np.concatenate(all_predictions)
    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, val_loader, test_loader = data_preparation()
#     model = ViT().to(device)
#     model = OCET(
#             num_classes=3,
#             dim=100,
#             depth=2,
#             heads=4,
#             dim_head=25,
#             mlp_dim=200,
#         )
    model = ViT(
        in_channels=1,
        embedding_dim=100,
        num_layers=2,
        num_heads=4,
        qkv_bias=False,
        mlp_ratio=2.0,
        dropout_rate=0.0,
        num_classes= 3,
    )
    model = model.to(device)
      
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0005)
    criterion = nn.CrossEntropyLoss(reduction='mean')
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), 
#                                  lr=0.001,#0.0005, 
# #                                  eps=1, 
#                                  weight_decay=1e-4)#, amsgrad=True)
    print(optimizer)
    epochs = 150
    train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs=epochs)
    compute_acc(test_loader)
    compute_metric(test_loader)



    


