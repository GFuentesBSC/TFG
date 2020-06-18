
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from horovodizer_helper_Torch import *
import horovod.torch as hvd
import torch.utils.data.distributed
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.manual_seed(0)
from torch.optim.lr_scheduler import StepLR
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
batch_size = 100
n_iters = 3000
num_epochs = (n_iters / (len(train_dataset) / batch_size))
num_epochs = int(num_epochs)
hvd_sampler_train_loader = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, sampler=hvd_sampler_train_loader)
hvd_sampler_test_loader = torch.utils.data.distributed.DistributedSampler(dataset=test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=hvd_sampler_test_loader)

class FeedforwardNeuralNetModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
input_dim = (28 * 28)
hidden_dim = 100
output_dim = 10
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
adapt_optimizer(optimizer, model)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
iter = 0
for epoch in range(num_epochs):
    scheduler.step()
    print('Epoch:', epoch, 'LR:', scheduler.get_lr())
    for (i, (images, labels)) in enumerate(train_loader):
        images = images.view((- 1), (28 * 28)).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        if ((iter % 500) == 0):
            correct = 0
            total = 0
            for (images, labels) in test_loader:
                images = images.view((- 1), (28 * 28))
                outputs = model(images)
                (_, predicted) = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = ((100 * correct) / total)
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
