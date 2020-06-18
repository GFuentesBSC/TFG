import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 32
num_classes = 10
epochs = 100
learning_rate = 0.001
momentum = 0.9

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("Downloading dataset...")
trainset = torchvision.datasets.CIFAR10(root='~/.keras/datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='~/.keras/datasets', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

net = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2), nn.Dropout(0.2),
      nn.Conv2d(64, 1024, kernel_size=3, padding=1), nn.ReLU(),
      nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2), nn.Dropout(0.2),
      nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(),
      nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(),
      nn.MaxPool2d(2), nn.Dropout(0.2),
      nn.Flatten(),
      nn.Linear(16384, 128), nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, 10)
      )
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    final_loss = round(train_loss/(batch_idx+1),2)
    final_acc  = round(100*correct/total,2)
    return final_loss, final_acc


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        final_loss = round(test_loss/(batch_idx+1),2)
        final_acc  = round(100*correct/total,2)
        return final_loss, final_acc

train_losses = []
train_accs   = []
test_losses  = []
test_accs    = []
print("Start training...")

for epoch in range(epochs):
    tr_l, tr_a = train(epoch)
    train_losses.append(tr_l)
    train_accs.append(tr_a)
    ts_l, ts_a = test(epoch)
    test_losses.append(ts_l)
    test_accs.append(ts_a)
    print(f"Epoch {epoch}: train_loss: {tr_l} train_acc: {tr_a} | test_loss: {ts_l} test_acc: {ts_a}")
