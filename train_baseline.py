import torch
import numpy as np
from tqdm import tqdm
from model import CNN
import matplotlib.pyplot as plt
from dataset import mnist_loader, visualize_dataset

NUM_EPOCHS = 10
VIS_DATA = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

train, validation, test = mnist_loader(val_split=0.2, batch_size=5)
if VIS_DATA: visualize_dataset([train, validation, test])

model = CNN().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

all_train_loss = list()
all_val_loss = list()
val_loss_min = np.Inf

for epoch in range(1, NUM_EPOCHS+1):
    print('-'*75)
    train_loss = 0.0
    val_loss = 0.0
    correct = 0    
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()*data.size(0)
        loss.backward()
        optimizer.step()
    
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(validation)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()*data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    
    train_loss = train_loss / len(train.dataset)
    all_train_loss.append(train_loss)
    val_loss = val_loss / len(validation.dataset)
    all_val_loss.append(val_loss)
    accuracy = correct / len(validation.dataset)
    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch , NUM_EPOCHS, train_loss, val_loss, accuracy))

    if val_loss < val_loss_min:
        val_loss_min = val_loss
        print("Saving Model State")
        torch.save(model.state_dict(), "models/mnist_baseline.sav")

model.load_state_dict(torch.load("models/mnist_baseline.sav"))
model.eval()
correct = 0
print('-'*75)
for batch_idx, (data, target) in tqdm(enumerate(test)):
    data, target = data.to(DEVICE), target.to(DEVICE)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()

print('Test accuracy {:.8f}'.format(correct/len(test.dataset)))