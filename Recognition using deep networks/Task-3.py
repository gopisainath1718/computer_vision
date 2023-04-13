# Gopisainath mamindlapalli and Krishna Prasath Senthil Kumaran

# import statements
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import os

# declaring some useful variables
n_epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class definitions
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x)

# class to transform the shape of the images
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# function to train the model
def train_network(epoch, network, greek_train, optimizer, train_counter, train_losses):

    network.train()
    for ids, (data, target) in enumerate(greek_train):
        # print(target)
        optimizer.zero_grad()
        output = network(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        # print(loss)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, ids * len(data), len(greek_train.dataset),100. * ids / len(greek_train), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (ids*5) + ((epoch-1)*len(greek_train.dataset)))
    
    return

# function to test the model
def test_network(network):

    
    dict = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    folder = "greek_test"
    images = []
    for i in os.listdir(folder):
        img = plt.imread(os.path.join(folder, i))
        img = img[:,:,0]
        img = img.reshape(1, 28, 28)
        if img is not None:
            images.append(img)
    images = torch.FloatTensor(images)
    
    with torch.no_grad():
        output = network(images.to(device))

    # plotting the output
    plt.figure()
    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')

        title = dict[output.data.max(1, keepdim=True)[1][i].item()]

        plt.title(f"Prediction: {title}")
        plt.xticks([])
        plt.yticks([])
    plt.show()



# main function
def main(argv):
    # handle any command line arguments in argv

    # loading greek letters
    training_set_path = "greek_train/greek_train"
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( training_set_path,
                                      transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,) ) ] ) ),
    batch_size = 5,
    shuffle = True )

    # changing the last layer of the network
    network = Net()
    filename = "model.sav"
    network = torch.load(filename, map_location=torch.device(device))

    for param in network.parameters():
        param.requires_grad = False

    in_number = network.fc2.in_features
    network.fc2 = nn.Linear(in_number, 3)
    print(network)
    network.to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.001)
    train_losses = []
    train_counter = []
    for epoch in range(n_epoch):
        # print(epoch)
        train_network(epoch, network, greek_train ,optimizer, train_counter, train_losses)
    
    # plotting the losses
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # function call
    test_network(network)

    return

if __name__ == "__main__":
    main(sys.argv)
