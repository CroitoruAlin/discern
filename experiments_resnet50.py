import os

import utils
import train
import test
import torch
import models.resnet as resnet
import torchvision

def train_models_cifar100(device):
    train_loader, val_loader = utils.get_train_val_loaders(val_size=4000, batch_size=200,
                                                           dataset=torchvision.datasets.CIFAR100)
    print(device)
    conv_layers = [5]

    model_c5 = resnet.ResNet50(conv_layers).to(device)
    optimizer = torch.optim.SGD(model_c5.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 200])
    train_model_c5 = train.Train(model_c5, device, train_loader, val_loader, optimizer, scheduler, pass_loss=False)
    train_model_c5.train('resnet50_c5', lmbd=[0.5], lmbd2=[0.5], start_epoch=10, num_epochs=250,
                             no_classes=100,
                             target_dist='batch_dist')


def test_models_cifar100(device):
    test_loader = utils.get_test_loader(batch_size=64, dataset=torchvision.datasets.CIFAR100)

    model = resnet.ResNet50([5]).to(device)
    model.load_state_dict(torch.load('./saved_models/resnet50_c5.pth'))
    test.test(model, device, test_loader, lmbd1=[0.5], lmbd2=[0.5], no_classes=100)

if __name__ == "__main__":
    os.makedirs("./saved_models",exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_models_cifar100(device)

    test_models_cifar100(device)

