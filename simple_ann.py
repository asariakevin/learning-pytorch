import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #inputs to hidden layer a Linear transformation
        self.fc1 = nn.Linear(784,256)

        #output layer,10 unit i.e one for each digit
        self.fc2 = nn.Linear(256,10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self,x):

        # hidden layer with sigmoid activation
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = F.softmax(self.fc2(x),dim=1) #don't forget the dim=1 argument

        return x


# create the network and look at it's representation
model = Network()
print(model)

# grab some data
dataiter = iter(trainloader)
images,labels = dataiter.next()

# resize images into a 1d vector
# new shape is (batch_size,color_channels,image_pixels)
images.resize_(64,1,784)

# forward pass through the network
image_index = 0
ps = model.forward(images[image_index,:])
print(ps)
