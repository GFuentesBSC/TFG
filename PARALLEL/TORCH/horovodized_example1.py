
from engine import train_one_epoch, evaluate
import utils
from horovodizer_helper_Torch import *
import horovod.torch as hvd
import torch
import torch.utils.data.distributed
hvd.init()
torch.cuda.set_device(hvd.local_rank())

def main():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    num_classes = 2
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:(- 50)])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[(- 50):])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    model = get_model_instance_segmentation(num_classes)
    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    adapt_optimizer(opt, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, opt, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
    print("That's it!")
