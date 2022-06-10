from tqdm import tqdm
import torch
import utils
from test import  test
import os
import torch.nn.functional as F
import sys

class Train:

    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, pass_loss=True):
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model
        self.best_accuracy = 0.0
        self.scheduler = scheduler
        self.pass_loss = pass_loss
        self.val_loader = val_loader
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def train_epoch(self, epoch_num, lmbd, lmbd2, start_epoch, target_dist, no_labels=10, filters_loss=utils.gini_index):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        epoch_loss = 0.0
        batch_num = 0.
        for (data, target) in pbar:
        # for (data, target) in self.train_loader:
            batch_num += 1.
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            y_pred, list_activation_maps = self.model(data)
            one_hot_target = torch.zeros(target.shape[0], no_labels)
            one_hot_target = one_hot_target.to(self.device)
            one_hot_target.scatter_(1, target.unsqueeze(1), 1.0)
            if epoch_num < start_epoch:
                lmbd = [0.] * len(lmbd)
                lmbd2 = [0.] * len(lmbd2)
            if target_dist == "batch_dist":

                new_target_dist = torch.bincount(target, weights=torch.ones(target.shape[0], dtype=torch.float32).to(self.device),
                                                 minlength=no_labels)
                new_target_dist /= float(target.shape[0])
            else:
                new_target_dist= target_dist
            loss, loss_ce, loss_gini_index, kl_div = utils.final_loss(list_activation_maps, one_hot_target, y_pred,
                                                                      target,
                                                                      lmbd, lmbd2=lmbd2,
                                                                      target_dist=new_target_dist,
                                                                      filters_loss=filters_loss)
            epoch_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f} Loss_CE={loss_ce.item()} Loss_gini={loss_gini_index.item()}, kl_div={kl_div.item()}')
        self.train_loss.append(epoch_loss / batch_num)
        self.train_acc.append(100 * correct / processed)
        # print(f"Epoch loss{epoch_loss/batch_num}, accuracy={100 * correct / processed:0.2f}")
        return epoch_loss, 100 * correct / processed

    def train(self, file_name, num_epochs=100, lmbd=None, lmbd2=None,
              start_epoch=10, target_dist=None, no_classes=10, filters_loss=utils.gini_index):
        if lmbd2 is None:
            lmbd2 = [0.2]
        if lmbd is None:
            lmbd = [0.8]
        for epoch in range(1, num_epochs + 1):
            print('Epoch', epoch)
            train_loss, _ = self.train_epoch(epoch, target_dist=target_dist, lmbd=lmbd, lmbd2=lmbd2,
                                             start_epoch=start_epoch, no_labels=no_classes, filters_loss=filters_loss)
            if self.pass_loss:
                self.scheduler.step(epoch)
            else:
                self.scheduler.step()
            test_loss, test_acc = test(self.model, self.device, self.val_loader, target_dist=target_dist,
                                       lmbd1=lmbd, lmbd2=lmbd2, no_classes=no_classes, filters_loss=filters_loss)
            self.val_acc.append(test_acc)
            self.val_loss.append(test_loss)
            if self.best_accuracy < test_acc:
                self.best_accuracy = test_acc
                torch.save(self.model.state_dict(), os.path.join(".", "saved_models", file_name + ".pth"))
                print('New best accuracy. Model Saved!')

    def get_history(self):
        return self.train_loss, self.train_acc, self.val_loss, self.val_acc
