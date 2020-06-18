import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import numpy
import h5py
import copy
from PIL import Image, ImageSequence
import cv2
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, Sentence, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BytePairEmbeddings
from flair.embeddings import BertEmbeddings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from typing import List, Union
from pathlib import Path
import seaborn as sn
import H5Dataset_modular as H5
from horovodizer_helper_Torch import *
import horovod.torch as hvd
import torch.utils.data.distributed
hvd.init()
torch.cuda.set_device(hvd.local_rank())
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def func():
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, sampler=hvd_sampler_train_loader)
    hvd_sampler_val_loader = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, sampler=hvd_sampler_val_loader)
    hvd_sampler_test_loader = torch.utils.data.distributed.DistributedSampler(dataset=test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, sampler=hvd_sampler_test_loader)
    dataloaders_dict = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    feature_extracting = True
    output_image_model = 300
    if ((full_model == 'combined') or (full_model == 'image') or (full_model == 'fusion')):
        if (image_model == 'resnet'):
            use_pretrained = True
            net = models.resnet50(pretrained=use_pretrained)
            if (image_training_type == 'feature_extract'):
                set_parameter_requires_grad(net, feature_extract)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, output_image_model)
        if (full_model == 'image'):
            PATH_best = (((((model_path + 'image/') + str(max_epochs)) + str(learning_rate)) + description) + '.pt')
            if ((image_model == 'mobilenetv2') or 'dense'):
                net.classifier = nn.Linear(num_ftrs, num_classes)
            elif (image_model == 'efficientnet'):
                net._fc = nn.Linear(num_ftrs, num_classes)
            elif (image_model == 'resnet'):
                net.fc = nn.Linear(num_ftrs, num_classes)
            model = net

    model.cuda()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-05)
    adapt_optimizer(optimizer, model)
    if (full_model == 'image'):
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    loss_values = []
    epoch_values = []
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(max_epochs):
        since = time.time()
        steps = 0
        correct = 0
        confusion_matrix = torch.zeros(num_classes, num_classes)
        for phase in ['train']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            batch_counter = 0
            for local_batch in dataloaders_dict[phase]:
                batchs_number = (len(dataloaders_dict[phase].dataset) / batch_size)
                batch_counter += 1
                if ((batch_counter % 1000) == 0):
                    print('batch counter', batch_counter)
                if (text_model == 'documentRNN'):
                    (ocr_text, labels) = (local_batch['ocr'], Variable(local_batch['class']))
                    tokenized_sentences = []
                    for sentence in ocr_text:
                        sentence_tokenized = Sentence(sentence, use_tokenizer=True)
                        tokenized_sentences.append(sentence_tokenized)
                        ocr_text = tokenized_sentences
                    labels.cuda()
                else:
                    (image, ocr_text, labels) = (Variable(local_batch['image']).cuda(), Variable(local_batch['ocr']).cuda(), Variable(local_batch['class']).cuda())
                steps += 1
                optimizer.zero_grad()
                if ((full_model == 'combined') or (full_model == 'fusion')):
                    outputs = model(ocr_text, image)
                elif (full_model == 'image'):
                    outputs = model(image)
                elif (full_model == 'text'):
                    outputs = model(ocr_text)
                (_, preds) = torch.max(outputs.data, 1)
                labels = labels.long()
                loss = criterion(outputs, labels)
                running_corrects += torch.sum((preds == labels.data))
                if (phase == 'val'):
                    for (t, p) in zip(labels.view((- 1)), preds.view((- 1))):
                        confusion_matrix[(t.long(), p.long())] += 1
                        if (t.long() == p.long()):
                            correct += 1
                if (phase == 'train'):
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            if (phase == 'val'):
                if ((epoch + 1) == max_epochs):
                    print(confusion_matrix)
                    print((confusion_matrix.diag() / confusion_matrix.sum(1)))
            epoch_loss = (running_loss / len(dataloaders_dict[phase].dataset))
            epoch_acc = (running_corrects.double() / len(dataloaders_dict[phase].dataset))
            if (phase == 'train'):
                train_loss = epoch_loss
            if (phase == 'val'):
                val_loss = epoch_loss
            if (phase == 'val'):
                if (full_model == 'image'):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step(epoch_loss)
            actual_lr = get_lr(optimizer)
            print('[Epoch {}/{}] {} lr: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs, phase, actual_lr, epoch_loss, epoch_acc))
            if ((phase == 'val') and (epoch_acc > best_acc)):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if (phase == 'val'):
                val_acc_history.append(epoch_acc)
        PATH_final = ((((((model_path + full_model) + '/') + str(max_epochs)) + str(learning_rate)) + description) + 'final.pt')
        print()
        time_elapsed = (time.time() - since)
        print('Training completed in {:.0f}m {:.0f}s'.format((time_elapsed // 60), (time_elapsed % 60)))
        print('Best val Acc: {:4f}'.format(best_acc))
    num_classes = 16
    print('Testing...')
    model.eval()
    correct = 0
    max_size = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for (i, local_batch) in enumerate(test_loader):
            if (text_model == 'documentRNN'):
                (ocr_text, labels) = (local_batch['ocr'], Variable(local_batch['class']))
                tokenized_sentences = []
                for sentence in ocr_text:
                    sentence_tokenized = Sentence(sentence, use_tokenizer=True)
                    tokenized_sentences.append(sentence_tokenized)
                    ocr_text = tokenized_sentences
                labels = labels.to(device)
            else:
                (image, ocr_text, labels) = (Variable(local_batch['image']), Variable(local_batch['ocr']), Variable(local_batch['class']))
                (image, ocr_text, labels) = (image.to(device), ocr_text.to(device), labels.to(device))
            if ((i % 400) == 0):
                print(i)
            if ((full_model == 'combined') or (full_model == 'fusion')):
                outputs = model(ocr_text, image)
            elif (full_model == 'image'):
                outputs = model(image)
            elif (full_model == 'text'):
                outputs = model(ocr_text)
            (_, preds) = torch.max(outputs.data, 1)
            for (t, p) in zip(labels.view((- 1)), preds.view((- 1))):
                confusion_matrix[(t.long(), p.long())] += 1
                if (t.long() == p.long()):
                    correct += 1
    print(confusion_matrix)
    print((confusion_matrix.diag() / confusion_matrix.sum(1)))
    print('Accuracy: ', (correct / 2482))
if (__name__ == '__main__'):
    print('inside --main--')
    flair.device = torch.device('cuda')
    print('GPU available', torch.cuda.is_available())
    device = torch.device('cuda:0')
    func()
