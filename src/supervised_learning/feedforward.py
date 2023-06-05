import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.supervised_learning.dataset import CustomDataset, Ta41Dataset
from src.supervised_learning.networks import SimpleFFNetwork
from src.utils import get_project_root
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd



# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 30 * 7 # 210, 30 jobs with 7 features
num_classes = 30+1 # 30 jobs + 1 for no-op (no operation)
num_epochs = 120
batch_size = 64
learning_rate = 0.001
dropout = 0.0

#################################
#   Load data
#################################

raw_dataset = Ta41Dataset.get_normal_dataset()

#train_len = int(len(train) * 0.9)
#valid_len = len(train) - train_len
train, test, valid = random_split(raw_dataset, [0.8, 0.1, 0.1])

#train_len = int(len(train)) * 0.9
#test_len = len(train) - train_len 
#train, test = random_split(train, [train_len, test_len])

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

model = SimpleFFNetwork(lr=learning_rate, n_actions=num_classes,
                                input_dims=input_size,
                                name=f"simple_ff_with_{dropout}_droput_{num_epochs}_epochs.pth",
                                dropout_value=dropout,
                                checkpoint_dir=os.path.join(get_project_root(), "models", "supervised")
)


def train(model, num_epoch, train_loader, valid_loader, print_every=1000):

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
            model.optimizer.zero_grad()
            # Forward pass
            outputs = model(states)
            # Find the loss
            loss = model.loss(outputs, actions)
            # Backward pass/Calculate gradients
            loss.backward()
            # Update the weights
            model.optimizer.step() 
            # Calculate the loss
            current_train_loss += loss.item()

        
        model.eval()    # Optional when not using Model Specific layer
        for i, (states, actions) in enumerate(valid_loader):
            states = states.reshape(-1, input_size).to(device) # Flatten?
            actions = actions.view(-1).to(device)   # Flatten?
            
            outputs = model(states)
            loss = model.loss(outputs, actions)
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
            model.save_checkpoint()
            #torch.save(model.state_dict(), 'saved_model.pth')

    print('Finished training.')
    plt.plot(range(num_epoch), train_loss_values, 'g', label='Training loss')
    plt.plot(range(num_epoch), valid_loss_values, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



train(model, num_epochs, train_loader, valid_loader,print_every=1)

#model.load_checkpoint()


y_pred = []
y_true = []

# test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i, (states, actions) in enumerate(test_loader):
        states = states.reshape(-1, input_size).to(device)
        actions = actions.to(device)
        outputs = model(states)

        #print(outputs)

        # value, index
        _, predictions = torch.max(outputs, 1)

        # flatten
        predictions = predictions.flatten().data.cpu().numpy()
        y_pred.extend(predictions) # Save Prediction

        actions = actions.flatten().data.cpu().numpy()
        y_true.extend(actions) # Save Truth

        #print(f"Predictions: {predictions}")
        #print(f"Actions: {actions}")

        n_samples += actions.shape[0]
        n_correct += (predictions == actions).sum().item()

print(f"n_correct: {n_correct}, n_samples: {n_samples}")

acc = 100.0 * n_correct / n_samples
print(f"accuracy = {acc}")

classes = range(0, num_classes)

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure()
sn.heatmap(df_cm, annot=True)
plt.show()