from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

train_dataset = 2
batch_size = 2
model = None
################################################################################################
# before hvdzr
################################################################################################
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
################################################################################################
# after hvdzr
################################################################################################
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset,
                                                                num_replicas=hvd.size(),
                                                                rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           sampler=train_sampler)

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
