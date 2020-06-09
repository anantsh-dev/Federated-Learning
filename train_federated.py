import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from model import CNN
import matplotlib.pyplot as plt
from dataset import mnist_loader, visualize_dataset
import copy

NUM_EPOCHS = 10
LOCAL_ITERS = 2
VIS_DATA = False
BATCH_SIZE = 5
NUM_CLIENTS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)


def FedAvg(params):
    global_params = dict(params[0])
    for key in global_params.keys():
        for param in params:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train(local_model, device, dataset, iters, criterion):
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), len(dataset)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            batch_loss += loss.item()*data.size(0)
            loss.backward()
            optimizer.step()
        train_loss+=batch_loss/len(dataset)
        print("tr", train_loss)
    return local_model.state_dict(), train_loss/iters


def test(model, dataloader, criterion):
    test_loss = 0.0
    correct = 0
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/BATCH_SIZE):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    
    return test_loss/len(dataloader.dataset), preds, accuracy


if __name__=="__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    logname = ('results/log_federated_' + str(NUM_EPOCHS))
    logging.basicConfig(filename=logname,level=logging.DEBUG)
    logger = logging.getLogger()
    
    train_data, validation_data, test_data = mnist_loader(val_split=0.2, batch_size=BATCH_SIZE)
    if VIS_DATA: visualize_dataset([train, validation, test])

    train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
    for batch_idx, (data,target) in enumerate(train_data):
        train_distributed_dataset[batch_idx % NUM_CLIENTS].append((data, target))

    global_model = CNN().to(DEVICE)
    global_params = global_model.state_dict()
    criterion = torch.nn.CrossEntropyLoss()

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    for epoch in range(1, NUM_EPOCHS+1):
        print("\nEpoch :", str(epoch))
        local_params, local_losses = [], []
        for idx in range(NUM_CLIENTS):
            param, loss = train(copy.deepcopy(global_model), DEVICE, \
                train_distributed_dataset[idx],LOCAL_ITERS,criterion)
            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(loss))
        
        global_params = FedAvg(local_params)
        global_model.load_state_dict(global_params)
        all_train_loss.append(sum(local_losses)/len(local_losses))
        
        val_loss, _, accuracy = test(global_model, validation_data, criterion)
        all_val_loss.append(val_loss)
        
        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'\
            .format(epoch , NUM_EPOCHS, all_train_loss[-1], val_loss, accuracy))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(global_model.state_dict(), "models/mnist_federated.sav")

    global_model.load_state_dict(torch.load("models/mnist_federated.sav"))
    test_loss, predictions, accuracy = test(global_model, test_data, criterion)
    logger.info('Test accuracy {:.8f}'.format(accuracy))