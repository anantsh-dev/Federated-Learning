from matplotlib import pyplot as plt
# with open("results/log_federated_fashion_mnist_10_3_2") as f:   
#     fed = f.readlines()
# with open("results/log_baseline_fashion_mnist_10") as f:    
#     base = f.readlines()

with open("results/log_federated_mnist_10_3_2") as f:   
    fed = f.readlines()
with open("results/log_baseline_mnist_10") as f:    
    base = f.readlines()


fed_val = [float(f.split(" ")[7].strip(",")) for f in fed if "Train" in f]
fed_train = [float(f.split(" ")[4].strip(",")) for f in fed if "Train" in f]
base_val = [float(b.split(" ")[7].strip(",")) for b in base if "Train" in b]
base_train = [float(b.split(" ")[4].strip(",")) for b in base if "Train" in b]

epochs = range(1,11)
plt.figure()
plt.plot(epochs, fed_train, 'g', label='Training loss')
plt.plot(epochs, fed_val, 'b', label='validation loss')
plt.title('Federated loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks()
# plt.ylim([0,3])
plt.ylim([0,1])


plt.figure()
plt.plot(epochs, base_train, 'g', label='Training loss')
plt.plot(epochs, base_val, 'b', label='validation loss')
plt.title('Baseline loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks()
# plt.ylim([0,3])
plt.ylim([0,1])
plt.show()
