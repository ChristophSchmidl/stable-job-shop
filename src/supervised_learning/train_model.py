import os
import torch
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from src.supervised_learning.dataset import Ta41Dataset
from src.supervised_learning.networks import SimpleFFNetwork
from src.utils import get_project_root
from src import config
import wandb



def train(
        data_filename = "30mins_tuned_policy/ta41/experiences_no-permutation_1000-episodes.npz", 
        instance_name = "ta41",
        data_desc = "no-permutation",
        use_dropout = False, 
        lr = 0.001, 
        num_classes = 30+1, 
        input_size = 30*7,
        num_epochs = 20, 
        time_limit = None
        ):
    '''
    - use_dropout: activate dropout or deactivate
    - lr: the learning rate
    - num_classes: number of classes, e.g., 30 jobs + 1 for no-op (no operation)
    - input_size: the flattened input size of an instance, e.g., 30 * 7 = 210 =  30 jobs with 7 features
    - num_epochs: number of epochs is only used when time_limit is None
    '''

    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.USE_WANDB:
        run = wandb.init(
            project=config.WANDB_PROJECT,
            notes=f"Supervised-learning on dataset {data_filename}. Using dropout: {use_dropout}. Instance name: {instance_name}. Data description: {data_desc}",
            group="Supervised-learning",
            job_type=f"{instance_name} with {data_desc}",
            tags=["supervised-learning", f"{instance_name}", f"{data_desc}" f"{time_limit}-seconds-time-limit", f"Dropout: {use_dropout}"]
        )
    
    wandb.config.update({
        "data_filename": data_filename, 
        "use_dropout": use_dropout,
        "lr": lr,
        "num_classes": num_classes,
        "input_size": input_size,
        "num_epochs": num_epochs,
        "time_limit_in_seconds": time_limit
    })

    if use_dropout:
        dropout = 0.5
    else:
        dropout = 0.0

    ta41_dataset = Ta41Dataset(file_name=data_filename)

    train, test, valid = random_split(ta41_dataset, [0.8, 0.0, 0.2])

    print(f"Length of train: {len(train)}")
    print(f"Length of test: {len(test)}")
    print(f"Length of valid: {len(valid)}")


    train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid, batch_size=64, shuffle=True)
    #test_loader = DataLoader(dataset=test, batch_size=64, shuffle=True)

    # instance_name: ta41
    # permutation mode: no-permutation, 20_percent_permutation, ....


    model = SimpleFFNetwork(lr=lr, n_actions=num_classes,
                                input_dims=input_size,
                                name=f"simple_ff_{instance_name}_with_{dropout}_droput_{time_limit}_sec-timelimit_{data_desc}.pth",
                                dropout_value=dropout,
                                checkpoint_dir=os.path.join(get_project_root(), "models", "supervised")
    )

    if config.USE_WANDB:
        wandb.watch(model, log_freq=100)

    since = datetime.now()
    

    best_loss = float('inf')
    best_acc = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = valid_loader

            running_loss = 0.0
            correct_predictions = 0
            y_true = []
            y_pred = []

            # Iterate over data
            for states, actions in data_loader:
                states = states.reshape(-1, input_size).to(device) # Flatten?
                actions = actions.view(-1).to(device)   # Flatten?
            
                # Clear the gradients
                model.optimizer.zero_grad()
                # Forward pass
                outputs = model(states)
                _, preds = torch.max(outputs, 1)
                # Find the loss
                loss = model.loss(outputs, actions)

                # Add to y_true and y_pred
                y_true.extend(actions.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                if phase == 'train':
                    # Backward pass/Calculate gradients
                    loss.backward()
                    # Update the weights
                    model.optimizer.step() 
                # Calculate the loss
                running_loss += loss.item() * states.size(0)
                correct_predictions += torch.sum(preds == actions.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = correct_predictions.double() / len(data_loader.dataset)
            epoch_f1 = f1_score(y_true, y_pred, average='macro')

            

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))

            # Log metrics to wandb
            if config.USE_WANDB:
                wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, f"{phase}_f1": epoch_f1})

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                model.save_checkpoint()

        # Check if time limit has passed
        if time_limit is not None and datetime.now() - since >= timedelta(seconds=time_limit):
            print(f"Training interrupted. Reached time limit of {time_limit} seconds.")
            break

    print('Best val loss: {:4f}, best val acc: {:4f}'.format(best_loss, best_acc))

    # load best model weights
    model.load_checkpoint()

    if config.USE_WANDB:
        wandb.save(model.checkpoint_file)

    wandb.finish()

    return model


