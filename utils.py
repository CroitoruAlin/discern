import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torchvision
from torch.utils.data import random_split
from torch import randperm
from torch import default_generator


def gini_index(activation_maps, y_target):
    mean_activation_maps = torch.mean(torch.abs(activation_maps), dim=(2, 3))
    sum_filters = torch.mm(torch.transpose(mean_activation_maps, 0, 1), y_target)
    count_classes = torch.sum(y_target, dim=0)
    sum_filters /= (count_classes + 1e-10)
    return gini_computation(sum_filters)


def gini_computation(sum_filters):
    prob_classes = sum_filters / (torch.sum(sum_filters, dim=1, keepdim=True) + (1e-10))
    loss = torch.mean(1 - torch.sum(prob_classes ** 2, dim=1))
    return loss, prob_classes


def entropy_comp(sum_filters):
    prob_classes = torch.softmax(sum_filters, dim=1)
    loss = torch.mean(-torch.sum(prob_classes * torch.log10(prob_classes), dim=1))
    return loss, prob_classes


def entropy(activation_maps, y_target):
    mean_activation_maps = torch.mean(activation_maps, dim=(2, 3))
    sum_filters = torch.mm(torch.transpose(mean_activation_maps, 0, 1), y_target)
    count_classes = torch.sum(y_target, dim=0)
    sum_filters /= (count_classes + 1e-10)
    return entropy_comp(sum_filters)


def kl_div(target_dist, prob_classes):
    divergence = 0.
    if target_dist is not None:
        prob_dist = torch.sum(prob_classes, dim=0)
        s = torch.sum(prob_dist, dim=0)
        prob_dist += (1e-10)
        prob_dist /= (s)
        divergence = F.kl_div(prob_dist.log(), target_dist, reduction='sum')
    return divergence


def final_loss(activation_maps, one_hot_target, y_pred, y_target, lmbd=None, lmbd2=None, lmbd3=1.,
               filters_loss=gini_index,
               target_dist=None):
    if lmbd2 is None:
        lmbd2 = [0.2]
    if lmbd is None:
        lmbd = [0.5]
    ce = torch.nn.CrossEntropyLoss()
    loss1 = ce(y_pred, y_target)
    losses = []
    kl = []

    if not isinstance(activation_maps, torch.Tensor) and len(activation_maps) > 0:
        for i in range(len(activation_maps)):
            l, prob_classes = filters_loss(activation_maps[i], one_hot_target)
            if i == 0:
                loss2 = l * lmbd[i]
                kl = kl_div(target_dist, prob_classes) * lmbd2[i]
            else:
                loss2 += l * lmbd[i]
                kl += kl_div(target_dist, prob_classes) * lmbd2[i]
        # losses = torch.stack(losses, dim=0)
        # loss2 = torch.sum(losses*lmbd, dim=0)
        loss2 /= len(activation_maps)
        div = kl / len(activation_maps)
    elif isinstance(activation_maps, torch.Tensor):
        loss2, prob_classes = filters_loss(activation_maps, one_hot_target)
        div = kl_div(target_dist, prob_classes)
    else:
        return loss1, torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])
    loss = loss1 * lmbd3 + loss2 + div
    return loss, loss1, loss2, div


def get_train_val_loaders(val_size=2500, batch_size=128, dataset=torchvision.datasets.CIFAR10,
                          transforms_list=None, apply_on_valid=False):
    if transforms_list is None:
        transforms_list = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]

    changes = transforms.Compose(transforms_list)
    val_transforms = [transforms.ToTensor()]
    if apply_on_valid:
        val_transforms = transforms_list
    val_changes = transforms.Compose(val_transforms)
    train_data = dataset(root='./data', train=True, download=True, transform=changes)
    val_data = dataset(root='./data', train=True, download=True, transform=val_changes)
    torch.manual_seed(33)
    train_size = len(train_data) - val_size
    indices = randperm(sum([train_size, val_size]), generator=default_generator).tolist()
    train_ds = Subset(train_data, indices[0: train_size])
    val_ds = Subset(val_data, indices[train_size: train_size + val_size])
    print(f'Train data size {len(train_ds)}, Validation data size {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_test_loader(batch_size=128, dataset=torchvision.datasets.CIFAR10):
    test_changes = transforms.Compose([transforms.ToTensor()])
    test_data = dataset(root='./data', train=False, download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def do_predict(model, data):
    return model.predict(data)
