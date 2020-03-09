from torch import nn

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #inputs to hidden layer a Linear transformation
        self.hidden = nn.Linear(784,256)

        #output layer,10 unit i.e one for each digit
        self.output = nn.Linear(256,10)

        #define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.sofmax = nn.Softmax(dim=1) # inorder to find softmax across columns

    def forward(self,x):

        # pass the input tensor through each of our operations

        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


# create the network and look at it's representation
model = Network()
print(model)
