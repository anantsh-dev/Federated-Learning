# Federated-Learning
Implemention of a CNN model in a federated learning setting. The dataset is distributed across a given number of clients and then the local model is trained for each client. The parameters from each client's model is then used to update the global model.

The experiment is performed on MNIST and FashionMNIST dataset. A simple CNN based model is used to train the neural network. The training dataset is split into 80% traning and 20% validation data, and the validation loss is used to save the best model. The results of model trained in a federated setting is compared with a simple (baseline) model trained centrally using the complete data. 

## Requirements
* Python3
* PyTorch
* TorchVision

## Directory Structure
* dataset.py - script used to load the dataset
* model.py - script used to initalize the CNN model
* train_baseline.py - script used to train the baseline model
* train_federated.py - script used to train the Federated learning model

## Training Options
Training Paramters are mentioned below and can be set inside train_baseline.py and train_federated.py
* NUM_EPOCHS : number of epochs to train the model
* BATCH_SIZE : Batch Size for the dataset
* NUM_CLIENTS : Number of Clients to simulate a federated setting (Only in train_federated.py)
* LOCAL_ITERS : Number of iterations performed by each client to update the local model (Only in train_federated.py)

## Execution Details:
Basline model can be trained using,
```
python3 train_baseline.py
```
Federated Learning model can be trained using.
```
python3 train_federated.py
```
## Results:
The model is trained for 10 epochs for both baseline and federated model (3 clientsa and 2 local iterations each) and the Test accuracy is reported as,

| Dataset        | Federated           | Baseline  |
| ------------- |:-------------:| -----:|
| MNIST       | 99.1% | 98.9% |
| FashionMNIST      | 91.3%      | 90.7%   |

The loss plots for all the models are displayed below,

### MNIST
 
Baseline Model            |  Federated Learning Model
:-------------------------:|:-------------------------:
![](https://github.com/ashar207/Federated-Learning/blob/master/results/mbase.png)  |  ![](https://github.com/ashar207/Federated-Learning/blob/master/results/mfed.png)

### Fashion MNIST
Baseline Model            |  Federated Learning Model
:-------------------------:|:-------------------------:
![](https://github.com/ashar207/Federated-Learning/blob/master/results/fbase.png)  |  ![](https://github.com/ashar207/Federated-Learning/blob/master/results/ffed.png)
