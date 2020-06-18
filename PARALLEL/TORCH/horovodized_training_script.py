from horovodizer_helper_Torch import *
from eff_utils import *
import horovod.torch as hvd
import torch
import torch.utils.data.distributed

def func():
    # number_gpus = torch.cuda.device_count()
    number_gpus = hvd.size()
    print('GPUs visible to PyTorch', number_gpus, 'GPUs')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int, required=True, help='Number of epochs in the training process. Should be an integer number')
    parser.add_argument('--eff_model', default='b0', type=str, required=True, help='EfficientNet model used b0, b1, b2, b3 or b4')
    parser.add_argument('--load_path', default='/gpfs/scratch/bsc31/bsc31275/', type=str, required=True, help='EfficientNet model used b0, b1, b2, b3 or b4')
    args = parser.parse_args()
    efficientnet_model = args.eff_model
    description = (((('eff' + efficientnet_model) + 'BT') + str(number_gpus)) + 'gpus_paper_exp_20_graph')
    max_epochs = args.epochs
    batch_size = ((16 * number_gpus) if (int(efficientnet_model[1]) < 3) else (8 * number_gpus))
    big_tobacco_classes = 16
    lr_multiplier = 0.2
    learning_rate = ((lr_multiplier * batch_size) / 256)
    number_workers = (4 * number_gpus)
    triangular_lr = True
    input_size = 384
    scratch_path = args.load_path
    save_path = (((((scratch_path + '/image_models/paper_experiments/') + str(max_epochs)) + str(learning_rate)) + description) + '.pt')
    hdf5_file = (scratch_path + '/BigTobacco_images_')
    print('batch size: ', batch_size)
    print('number of workers: ', number_workers)
    print('max_epochs', max_epochs)
    print('efficientnet_model', efficientnet_model)
    documents_datasets = {x: H5.H5Dataset(path=str(((hdf5_file + x) + '.hdf5')), data_transforms=data_transforms[x], phase=x) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(documents_datasets[x], batch_size=(batch_size if (x == 'train') else int((batch_size / (number_gpus * 2)))), shuffle=False, num_workers=number_workers, pin_memory=True, sampler=torch.utils.data.distributed.DistributedSampler(dataset=documents_datasets[x], num_replicas=hvd.size(), rank=hvd.rank())) for x in ['train', 'val']}
    feature_extracting = True
    net = EfficientNet.from_pretrained(('efficientnet-' + efficientnet_model), num_classes=1000)
    num_ftrs = net._fc.in_features
    net._fc = nn.Linear(num_ftrs, big_tobacco_classes)
    model = net
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-05)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    adapt_optimizer(optimizer, model)
    batches_per_epoch = (len(dataloaders_dict['train'].dataset) / batch_size)
    if (triangular_lr == True):
        scheduler = SlantedTriangular(optimizer, num_epochs=max_epochs, num_steps_per_epoch=batches_per_epoch, gradual_unfreezing=False)
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
    loss_values = []
    epoch_values = []
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    initial_train_time = 0
    initial_val_time = 0
    for epoch in range(max_epochs):
        results_file = []
        since = time.time()
        correct = 0
        for phase in ['train', 'val']:
            if (phase == 'train'):
                initial_train_time = time.time()
                if (epoch >= 1):
                    end_validation_time = (time.time() - initial_val_time)
                    print('Validation time: ', end_validation_time)
                model.train()
            else:
                end_train_time = (time.time() - initial_train_time)
                print('Training time: ', end_train_time)
                initial_val_time = time.time()
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            batch_counter = 0
            for local_batch in dataloaders_dict[phase]:
                batchs_number = (len(dataloaders_dict[phase].dataset) / batch_size)
                batch_counter += 1
                (image, labels) = (Variable(local_batch['image']), Variable(local_batch['class']))
                (image, labels) = (image.cuda(), labels.cuda())
                optimizer.zero_grad()
                outputs = model(image)
                (_, preds) = torch.max(outputs.data, 1)
                labels = labels.long()
                loss = criterion(outputs, labels)
                running_corrects += torch.sum((preds == labels.data))
                if (phase == 'train'):
                    loss.backward()
                    optimizer.step()
                    if (triangular_lr == True):
                        scheduler.step_batch()
                running_loss += loss.item()
            epoch_loss = (running_loss / len(dataloaders_dict[phase].dataset))
            epoch_acc = (running_corrects.double() / len(dataloaders_dict[phase].dataset))
            if (phase == 'train'):
                train_loss = epoch_loss
            if (phase == 'val'):
                val_loss = epoch_loss
            if (phase == 'val'):
                if (triangular_lr == True):
                    scheduler.step()
                else:
                    scheduler.step(epoch_loss)
            actual_lr = get_lr(optimizer)
            print('[Epoch {}/{}] {} lr: {:.4f}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, max_epochs, phase, actual_lr, epoch_loss, epoch_acc))
            if ((phase == 'val') and (epoch_acc > best_acc)):
                best_acc = epoch_acc
        print()
        time_elapsed = (time.time() - since)
        print('Training completed in {:.0f}m {:.0f}s'.format((time_elapsed // 60), (time_elapsed % 60)))
        print('Best val Acc: {:4f}'.format(best_acc))
        print()
    h5_dataset_test = H5.H5Dataset(path=str(((hdf5_file + 'test') + '.hdf5')), data_transforms=data_transforms['test'], phase='test')
    hvd_sampler_dataloader_test = torch.utils.data.distributed.DistributedSampler(dataset=h5_dataset_test, num_replicas=hvd.size(), rank=hvd.rank())
    dataloader_test = DataLoader(h5_dataset_test, batch_size=int((batch_size / (number_gpus * 2))), shuffle=False, num_workers=4, sampler=hvd_sampler_dataloader_test)
    print('Testing...')
    model.eval()
    correct = 0
    max_size = 0
    confusion_matrix = torch.zeros(big_tobacco_classes, big_tobacco_classes)
    with torch.no_grad():
        for (i, local_batch) in enumerate(dataloader_test):
            (image, labels) = (Variable(local_batch['image']), Variable(local_batch['class']))
            (image, labels) = (image.cuda(), labels.cuda())
            outputs = model(image)
            (_, preds) = torch.max(outputs.data, 1)
            for (t, p) in zip(labels.view((- 1)), preds.view((- 1))):
                confusion_matrix[(t.long(), p.long())] += 1
                if (t.long() == p.long()):
                    correct += 1
    print(confusion_matrix)
    print((confusion_matrix.diag() / confusion_matrix.sum(1)))
    print('Accuracy: ', (correct / len(dataloader_test.dataset)))
if (__name__ == '__main__'):
    # torch.multiprocessing.set_start_method('spawn', force='True')
    # print('GPU available', torch.cuda.is_available())
    # device = torch.device('cuda:0')
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    func()
