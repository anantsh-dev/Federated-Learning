import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from model import CNN
import matplotlib.pyplot as plt
from dataset import mnist_loader, visualize_dataset

NUM_EPOCHS = 10
VIS_DATA = False
BATCH_SIZE = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)


def train(model, device, dataloader, criterion, optimizer):
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/BATCH_SIZE):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()*data.size(0)
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader.dataset)


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
    
    logname = ('results/log_baseline_' + str(NUM_EPOCHS))
    logging.basicConfig(filename=logname,level=logging.DEBUG)
    logger = logging.getLogger()
    
    train_data, validation_data, test_data = mnist_loader(val_split=0.2, batch_size=BATCH_SIZE)
    if VIS_DATA: visualize_dataset([train, validation, test])

    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    for epoch in range(1, NUM_EPOCHS+1):
        print("\nEpoch :", str(epoch))
        train_loss = train(model, DEVICE, train_data, criterion, optimizer)
        val_loss, _, accuracy = test(model, validation_data, criterion)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch , NUM_EPOCHS, train_loss, val_loss, accuracy))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(model.state_dict(), "models/mnist_baseline.sav")

    model.load_state_dict(torch.load("models/mnist_baseline.sav"))
    test_loss, predictions, accuracy = test(model, test_data, criterion)
    logger.info('Test accuracy {:.8f}'.format(accuracy))