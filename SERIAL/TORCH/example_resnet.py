from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import tensorboardX
import os
import math
from tqdm import tqdm


hvd.init()

torch.cuda.set_device(hvd.local_rank())

# cudnn.benchmark = True

# Horovod: limit # of CPU threads to be used per worker.
# torch.set_num_threads(4)

# kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
# train_dataset = \
#     datasets.ImageFolder(args.train_dir,
#                          transform=transforms.Compose([
#                              transforms.RandomResizedCrop(224),
#                              transforms.RandomHorizontalFlip(),
#                              transforms.ToTensor(),
#                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                   std=[0.229, 0.224, 0.225])
#                          ]))
description = 'flair'
max_epochs = 10
batch_size = 128
num_classes = 16
full_model = 'image'
image_model = 'resnet'
image_training_type = 'finetuning'
text_model = 'cnn'
combined_embeddings = 'stack'
model_path = './resources/taggers/small_tobacco/'
learning_rate = 0.008
hdf5_file_val = '/gpfs/scratch/bsc31/bsc31275/BigTobacco_images_val.hdf5'
hdf5_file_test = '/gpfs/scratch/bsc31/bsc31275/BigTobacco_images_test.hdf5'
hdf5_file_train = '/gpfs/scratch/bsc31/bsc31275/BigTobacco_images_train.hdf5'
if (combined_embeddings == 'documentpool'):
    text_model = 'linear'
fast_embedding = WordEmbeddings('en')
embeddings_models_list = [fast_embedding]
embedding_model = StackedEmbeddings(embeddings=embeddings_models_list)
total_embedding_length = embedding_model.embedding_length
__all__ = ['MobileNetV2', 'mobilenetv2_19']
input_size = 224
data_transforms = {
    'train': transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}
train_dataset = H5.H5Dataset(path=hdf5_file_train, data_transforms=data_transforms['train'], embedding_model=embedding_model, embeddings_combination=combined_embeddings, type_model=full_model, phase='train')
val_dataset = H5.H5Dataset(path=hdf5_file_val, data_transforms=data_transforms['val'], embedding_model=embedding_model, embeddings_combination=combined_embeddings, type_model=full_model, phase='val')
test_dataset = H5.H5Dataset(path=hdf5_file_test, data_transforms=data_transforms['test'], embedding_model=embedding_model, embeddings_combination=combined_embeddings, type_model=full_model, phase='test')

hvd_sampler_train_loader = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=hvd_sampler_train_loader, **kwargs)

hvd_sampler_val_loader = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=hvd_sampler_val_loader, **kwargs)

hvd_sampler_test_loader = torch.utils.data.distributed.DistributedSampler(dataset=test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, sampler=hvd_sampler_test_loader)


# Set up standard ResNet-50 model.
model = models.resnet50(pretrained = True)

model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(learning_rate * hvd.size()),
                      momentum=0.9, weight_decay=4e-05)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())


# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

print("DONE")
