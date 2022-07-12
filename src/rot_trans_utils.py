import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.in_channels = 1
        self.input_size = 28
        self.conv1 = nn.Conv2d(self.in_channels, 6, 5,
                               padding=2 if self.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RotTransformer(nn.Module):
    '''
        Define a model to do the rotations - it has a meta-learnable 
        parameter theta that represents the rotation angle in radians
    '''
    def __init__(self, device):
        super(RotTransformer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor([0.]))
        # print("theta shape", self.theta.shape)
        self.device = device

    # Rotation transformer network forward function
    def rot(self, x):
        rot = torch.cat([torch.cat([torch.cos(self.theta), -torch.sin(self.theta), torch.tensor([0.], device=self.device)]),
                         torch.cat([torch.sin(self.theta), torch.cos(self.theta), torch.tensor([0.], device=self.device)])])
        grid = F.affine_grid(rot.expand([x.size()[0], 6]).view(-1, 2, 3), x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        return self.rot(x)

def get_device():
    if torch.cuda.is_available():  # checks whether a cuda gpu is available
        device = torch.cuda.current_device()
        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU
    return device

def get_dataloaders():
    transform_basic = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_rotate = transforms.Compose([
        transforms.RandomRotation([30, 30]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    train_set = datasets.MNIST(
        'data', train=True, transform=transform_basic, target_transform=None, download=True)
    train_set_rotated = datasets.MNIST(
        'data', train=True, transform=transform_rotate, target_transform=None, download=True)

    train_basic_indices = range(40000)
    train_test_basic_indices = range(40000, 50000)
    val_rotate_indices = range(50000, 60000)

    train_basic_set = torch.utils.data.Subset(train_set, train_basic_indices)
    train_test_basic_set = torch.utils.data.Subset(train_set, train_test_basic_indices)
    val_rotate_set = torch.utils.data.Subset(
        train_set_rotated, val_rotate_indices)
    test_set = datasets.MNIST(
    'data', train=False, transform=transform_rotate, target_transform=None, download=True)


    batch_size = 128

    train_basic_set_loader = torch.utils.data.DataLoader(
        train_basic_set, batch_size=batch_size, shuffle=True)
    train_test_basic_set_loader = torch.utils.data.DataLoader(
        train_test_basic_set, batch_size=batch_size, shuffle=True)
    val_rotate_set_loader = torch.utils.data.DataLoader(
        val_rotate_set, batch_size=batch_size, shuffle=True)
    test_set_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True)

    return train_basic_set_loader, train_test_basic_set_loader, val_rotate_set_loader, test_set_loader

def rot_img(x, theta, device):
    rot = torch.cat([torch.cat([torch.cos(theta), -torch.sin(theta), torch.tensor([0.], device=device)]),
                        torch.cat([torch.sin(theta), torch.cos(theta), torch.tensor([0.], device=device)])])
    grid = F.affine_grid(rot.expand([x.size()[0], 6]).view(-1, 2, 3), x.size())
    x = F.grid_sample(x, grid)
    return x

def test_classification_net(data_loader, model, device):
    '''
    This function reports classification accuracy over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            _, predictions = torch.max(softmax, dim=1)

            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return 100 * accuracy


def test_classification_net_rot(data_loader, model, device, angle=0.0):
    '''
    This function reports classification accuracy over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            if angle != 0.0:
                data = rot_img(data, angle, device)
            label = label.to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            _, predictions = torch.max(softmax, dim=1)

            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return 100 * accuracy