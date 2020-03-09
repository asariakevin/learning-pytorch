import torch.nn.functional as F
from torch import nn

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #inputs to hidden layer a Linear transformation
        self.hidden = nn.Linear(784,256)

        #output layer,10 unit i.e one for each digit
        self.output = nn.Linear(256,10)


    def forward(self,x):

        # hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x),dim=1)

        return x


# create the network and look at it's representation
model = Network()
print(model)
