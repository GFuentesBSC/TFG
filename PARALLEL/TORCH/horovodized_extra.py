import torch
from horovodizer_helper_Torch import *
import horovod.torch as hvd
import torch.utils.data.distributed
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.manual_seed(args.seed)
torch.set_num_threads(1)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=2, rank=0)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=2, rank=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)
optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=args.momentum)
adapt_optimizer(optimizer, model)

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    for (batch_idx, (data, target)) in enumerate(train_loader):
        if args.cuda:
            (data, target) = (data.cuda(), target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if ((batch_idx % args.log_interval) == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx * len(data)), len(train_sampler), ((100.0 * batch_idx) / len(train_loader)), loss.item()))

def metric_average(val, name):
    return 1

def test():
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for (data, target) in test_loader:
        if args.cuda:
            (data, target) = (data.cuda(), target.cuda())
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
    if (0 == 0):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, (100.0 * test_accuracy)))
