import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import horovod.torch as hvd


def hvd_metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def hvd_adapt_optimizer(optimizer, model):
    optimizer.param_groups[0]["lr"] *= hvd.size()
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    return optimizer


hvd.init()
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(((16 * 5) * 5), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view((-1), ((16 * 5) * 5))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
hvd_sampler_trainloader = torch.utils.data.distributed.DistributedSampler(
    dataset=trainset, num_replicas=hvd.size(), rank=hvd.rank()
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2, sampler=hvd_sampler_trainloader
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
hvd_sampler_testloader = torch.utils.data.distributed.DistributedSampler(
    dataset=testset, num_replicas=hvd.size(), rank=hvd.rank()
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2, sampler=hvd_sampler_testloader
)
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
net = Net()
net.cuda()
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
hvd_adapt_optimizer(optimizer, net)
for epoch in range(2):
    running_loss = 0.0
    for (i, data) in enumerate(trainloader, 0):
        (inputs, labels) = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i % 2000) == 1999:
            print(
                ("[%d, %5d] loss: %.3f" % ((epoch + 1), (i + 1), (running_loss / 2000)))
            )
            running_loss = 0.0
print("Finished Training")
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        (images, labels) = data
        outputs = net(images)
        (_, predicted) = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(
    (
        "Accuracy of the network on the 10000 test images: %d %%"
        % ((100 * correct) / total)
    )
)
class_correct = list((0.0 for i in range(10)))
class_total = list((0.0 for i in range(10)))
with torch.no_grad():
    for data in testloader:
        (images, labels) = data
        outputs = net(images)
        (_, predicted) = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print(
        (
            "Accuracy of %5s : %2d %%"
            % (classes[i], ((100 * class_correct[i]) / class_total[i]))
        )
    )
