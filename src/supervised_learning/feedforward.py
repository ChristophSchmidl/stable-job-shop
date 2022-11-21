import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.supervised_learning.dataset import CustomDataset, Ta41Dataset


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 30 * 7 # 210, 30 jobs with 7 features
hidden_size = 100
num_classes = 30+1 # 30 jobs + 1 for no-op (no operation)
num_epochs = 10
batch_size = 64
learning_rate = 0.001

#################################
#   Load data
#################################

train = Ta41Dataset.get_transposed_dataset(n_swaps=1)
train_len = int(len(train) * 0.9)
valid_len = len(train) - train_len

train, valid = random_split(train,[train_len, valid_len])

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)


# fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes) 
        )
    
    def forward(self, x):
        x = x.float() # Can I get rid of this using PyTorch transforms?
        return self.base(x)

model = NeuralNet(input_size, hidden_size, num_classes).to(device) # move the model to the device (GPU or CPU)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss() in one single class
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer


def train(model, num_epoch, train_loader, valid_loader, criterion, optimizer, print_every=1000):

    min_valid_loss = np.inf
    n_total_steps_train = len(train_loader)
    n_total_steps_valid = len(valid_loader)
    train_loss_values = []
    valid_loss_values = []

    print("Training started.")

    for epoch in tqdm(range(num_epoch)):
        current_train_loss = 0.0
        current_valid_loss = 0.0

        model.train() 
        for i, (states, actions) in enumerate(train_loader):
            states = states.reshape(-1, input_size).to(device) # Flatten?
            actions = actions.view(-1).to(device)   # Flatten?
            
            # Clear the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(states)
            # Find the loss
            loss = criterion(outputs, actions)
            # Backward pass/Calculate gradients
            loss.backward()
            # Update the weights
            optimizer.step() 
            # Calculate the loss
            current_train_loss += loss.item()

        
        model.eval()    # Optional when not using Model Specific layer
        for i, (states, actions) in enumerate(valid_loader):
            states = states.reshape(-1, input_size).to(device) # Flatten?
            actions = actions.view(-1).to(device)   # Flatten?
            
            outputs = model(states)
            loss = criterion(outputs, actions)
            current_valid_loss += loss.item()
        
        train_loss_values.append(current_train_loss / n_total_steps_train)
        valid_loss_values.append(current_valid_loss / n_total_steps_valid)

        # print every n steps
        if (i+1) % print_every == 0:
            print(f'Epoch {epoch+1}/{num_epochs} \t\t Training loss: {(current_train_loss /  n_total_steps_train):.4f} \t\t Validation loss: {(current_valid_loss / n_total_steps_valid):.4f}')

        if min_valid_loss > current_valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{current_valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = current_valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')

    print('Finished training.')
    plt.plot(range(num_epoch), train_loss_values, 'g', label='Training loss')
    plt.plot(range(num_epoch), valid_loss_values, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



train(model, num_epochs, train_loader, valid_loader, criterion, optimizer, print_every=1)



# test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f"accuracy = {acc}")