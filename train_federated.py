#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script used to train the Federated Model
@author : Anant
"""
import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from model import CNN
import matplotlib.pyplot as plt
from dataset import load_dataset, visualize_dataset
import copy

NUM_EPOCHS = 10
LOCAL_ITERS = 2
VIS_DATA = False
BATCH_SIZE = 5
NUM_CLIENTS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

DATASET = "fashion_mnist"
# DATASET = "mnist"

def FedAvg(params):
    """
    Average the paramters from each client to update the global model
    :param params: list of paramters from each client's model
    :return global_params: average of paramters from each client
    """
    global_params = dict(params[0])
    for key in global_params.keys():
        for param in params:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train(local_model, device, dataset, iters):
    """
    Trains a local model for a given client
    :param local_model: a copy of global CNN model required for training
    :param device: the device used to train the model - GPU/CPU
    :param dataset: training dataset used to train the model
    :return local_params: parameters from the trained model from the client
    :return train_loss: training loss for the current epoch
    """    
    #criterion and  optimzer for training the local models
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()

    #Iterate for the given number of Client Iterations
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
            data, target = data.to(device), target.to(device)
            #set gradients to zero            
            optimizer.zero_grad()
            #Get output prediction from the Client model
            output = local_model(data)
            #Computer loss
            loss = criterion(output, target)
            batch_loss += loss.item()*data.size(0)
            #Collect new set of gradients
            loss.backward()
            #Update local model
            optimizer.step()
        #add loss for each iteration
        train_loss+=(batch_loss/len(dataset))
    return local_model.state_dict(), train_loss/iters


def test(model, dataloader):
    """
    Tests the Federated global model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """    
    test_loss = 0.0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
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
    
    #Initialize a logger to log epoch results
    logname = ('results/log_federated_' + DATASET + "_" + str(NUM_EPOCHS))
    logging.basicConfig(filename=logname,level=logging.DEBUG)
    logger = logging.getLogger()
    
    
    #get data
    train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    if VIS_DATA: visualize_dataset([train, validation, test])

    #distribute the trainning data across clients
    train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
    for batch_idx, (data,target) in enumerate(train_data):
        train_distributed_dataset[batch_idx % NUM_CLIENTS].append((data, target))
    
    #get model
    global_model = CNN().to(DEVICE)
    global_model.train()
    global_params = global_model.state_dict()

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf
    
    #Train the model for given number of epochs
    for epoch in range(1, NUM_EPOCHS+1):
        print("\nEpoch :", str(epoch))
        local_params, local_losses = [], []
        #Send a copy of global model to each client
        for idx in range(NUM_CLIENTS):
            #Perform training on client side and get the parameters
            param, loss = train(copy.deepcopy(global_model), DEVICE, train_distributed_dataset[idx],LOCAL_ITERS)
            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(loss))
        
        #Federated Average for the paramters from each client
        global_params = FedAvg(local_params)
        #Update the global model
        global_model.load_state_dict(global_params)
        all_train_loss.append(sum(local_losses)/len(local_losses))
        
        #Test the global model
        val_loss, _, accuracy = test(global_model, validation_data)
        all_val_loss.append(val_loss)
        
        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'\
            .format(epoch , NUM_EPOCHS, all_train_loss[-1], val_loss, accuracy))
        
        #if validation loss decreases, save the model
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(global_model.state_dict(), "models/mnist_federated.sav")
    
    #load the best model from training
    global_model.load_state_dict(torch.load("models/mnist_federated.sav"))
    #test the model using test data
    test_loss, predictions, accuracy = test(global_model, test_data)
    logger.info('Test accuracy {:.8f}'.format(accuracy))