#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script used to train the Baseline Model
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

NUM_EPOCHS = 10
VIS_DATA = False
BATCH_SIZE = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)
# DATASET = "fashion_mnist"
DATASET = "mnist"

def train(model, device, dataloader, criterion, optimizer):
    """
    Trains a baseline model for the given dataset
    :param model: a CNN model required for training
    :param device: the device used to train the model - GPU/CPU
    :param dataloader: training data iterator used to train the model
    :param criterion: criterion used to calculate the traninig loss
    :param optimzer: Optimzer used to update the model parameters using backpropagation
    :return train_loss: training loss for the current epoch
    """
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/BATCH_SIZE):
        data, target = data.to(device), target.to(device)
        #set gradients to zero
        optimizer.zero_grad()
        #Get output prediction from the model
        output = model(data)
        #Computer loss
        loss = criterion(output, target)
        train_loss += loss.item()*data.size(0)
        #Collect new set of gradients
        loss.backward()
        #Upadate the model
        optimizer.step()

    return train_loss / len(dataloader.dataset)


def test(model, dataloader, criterion):
    """
    Tests the baseline model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :param criterion: criterion used to calculate the test loss
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """
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
    
    #Initialize a logger to log epoch results
    logname = ('results/log_baseline_' + DATASET + "_" + str(NUM_EPOCHS))
    logging.basicConfig(filename=logname,level=logging.DEBUG)
    logger = logging.getLogger()
    
    #get data
    train_data, validation_data, test_data = load_dataset(val_split=0.2, batch_size=BATCH_SIZE, dataset=DATASET)
    if VIS_DATA: visualize_dataset([train, validation, test])

    #get model and define criterion for loss and optimizer for model update
    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    #Train the model for given number of epochs
    for epoch in range(1, NUM_EPOCHS+1):
        print("\nEpoch :", str(epoch))
        #train using training data
        train_loss = train(model, DEVICE, train_data, criterion, optimizer)
        #test on validation data
        val_loss, _, accuracy = test(model, validation_data, criterion)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch , NUM_EPOCHS, train_loss, val_loss, accuracy))
        #if validation loss decreases, save the model
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(model.state_dict(), "models/mnist_baseline.sav")

    #load the best model from training
    model.load_state_dict(torch.load("models/mnist_baseline.sav"))
    #test the model using test data
    test_loss, predictions, accuracy = test(model, test_data, criterion)
    logger.info('Test accuracy {:.8f}'.format(accuracy))