import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #inputs to hidden layer a Linear transformation
        self.fc1 = nn.Linear(784,256)

        #output layer,10 unit i.e one for each digit
        self.fc2 = nn.Linear(256,10)


    def forward(self,x):

        # hidden layer with sigmoid activation
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1) #don't forget the dim=1 argument

        return x


# create the network and look at it's representation
model = Network()
print(model)
