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


# hyperparameters for our network
input_size = 784
hidden_size = [128,64]
output_size = 10

# create the network using nn.Sequential
model = nn.Sequential(
            nn.Linear(input_size,hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0],hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1],output_size),
            nn.Softmax(dim=1)
        )

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
