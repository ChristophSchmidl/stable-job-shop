import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.layer1 = nn.Linear(128, 32) # 128 features/input, 32 neurons/outputs
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1) # 1 actions/outputs
        '''
        self.base = nn.Sequential(
            nn.Linear(128, 32) # 128 features/input, 32 neurons/outputs
            nn.Linear(32, 16)
            nn.Linear(16, 1)
        )


    def forward(self, features):
        # (32, 128) -> 32 batch_size, 128 features/input
        '''
        x = self.layer1(features)
        x = self.layer2(x)
        x = self.layer3(x)
        '''
        x = self.base(features)
        # (32, 1)
        return x


if __name__ == "__main__":
    model = Model()
    features = torch.rand((2, 128)) # random features (2 rows, 128 columns/features)
    #print(features)
    print(model(features))
    print(features.device)

    #features = features.to("cuda") # move features to GPU
    #model = Model()
    #model.to("cuda") # move model to GPU