import torch
import utils


def test(model, device, test_loader, lmbd1, lmbd2, target_dist=None, no_classes=10, filters_loss=utils.gini_index):
    model.eval()
    test_loss = 0
    l_ce = 0
    l_gini = 0
    l_kl = 0
    correct = 0
    batch_num = 0.
    with torch.no_grad():
        for data, target in test_loader:
            batch_num += 1.
            data, target = data.to(device), target.to(device)
            output, list_activation_maps = model(data)
            one_hot_target = torch.zeros(target.shape[0], no_classes)
            one_hot_target = one_hot_target.to(device)
            one_hot_target.scatter_(1, target.unsqueeze(1), 1.0)
            if target_dist == "batch_dist":
                new_target_dist = torch.bincount(target,
                                                 weights=torch.ones(target.shape, dtype=torch.float32).to(device),
                                                 minlength=no_classes)
                new_target_dist /= float(target.shape[0])
            else:
                new_target_dist = target_dist
            l, l1, l2, l3 = utils.final_loss(list_activation_maps, one_hot_target, output, target,
                                             target_dist=new_target_dist, lmbd=lmbd1, lmbd2=lmbd2,
                                             filters_loss=filters_loss)
            test_loss += l
            l_ce += l1
            l_gini += l2
            l_kl += l3
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= batch_num

    print('\nTest set: Average loss: {:.4f},Average loss_ce: {},Average loss_gini: {}, Average kl div: {}'
          ' Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, l_ce / batch_num, l_gini / batch_num, l_kl / batch_num, correct,
        len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)

