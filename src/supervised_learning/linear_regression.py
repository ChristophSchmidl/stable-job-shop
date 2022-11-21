import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from src.supervised_learning.dataset import CustomDataset



data, targets = make_classification(n_samples=1000)
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, stratify=targets)

train_dataset = CustomDataset(train_data, train_targets)
test_dataset = CustomDataset(test_data, test_targets)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = lambda x, w, b: torch.matmul(x, w) + b # linear regression model

#print(train_data.shape) # (750, 20) 750 samples, 20 features

W = torch.randn(20, 1, requires_grad=True) # 20 features, 1 output
b = torch.randn(1, requires_grad=True) # 1 output
learning_rate = 0.001   # how much to update the weights at each step

for epoch in range(10): # 10 epochs
    epoch_loss = 0
    counter = 0
    for data in train_loader:
        x_train = data["x"]
        y_train = data["y"]

        # This is probably not necessary because PyTorch automatically does this 
        if W.grad is not None:
            W.grad.zero_()

        output = model(x_train, W, b) # forward pass
        loss = torch.mean( (y_train.view(-1) - output.view(-1))**2 )   # mean squared error loss
        epoch_loss = epoch_loss + loss.item() # accumulate loss for this epoch
        loss.backward() # backward pass/calculate the gradients

        with torch.no_grad():
            W = W - learning_rate * W.grad # update the weights
            b = b - learning_rate * b.grad # update the bias

        W.requires_grad_(True) # _ means in-place operation
        b.requires_grad_(True) # _ means in-place operation
        counter += 1

    print(epoch, epoch_loss/counter)

outputs = []
labels = []

with torch.no_grad():
    for data in test_loader:
        x_test = data["x"]
        y_test = data["y"]

        output = model(x_test, W, b) # forward pass
        labels.append(y_test)
        outputs.append(output)

print(f"ROC score: {metrics.roc_auc_score( torch.cat(labels).view(-1), torch.cat(outputs).view(-1))}")



