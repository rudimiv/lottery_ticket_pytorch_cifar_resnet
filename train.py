from torchvision import datasets, transforms
from my_resnet import CifarResNet20
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pytorch_model_summary import summary
import torch.nn.utils.prune as prune
import math


'''class WarmUPScheduler():
    def __init__(
            self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)'''

def train_epoch(model, data_loader, criterion, optimizer, device, scheduler=None):
    # switch to train mode
    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        train_loss += loss

        correct += (target == output.argmax(dim=1)).sum().item()
        total += target.size(0)

        loss.backward()
        optimizer.step()
        if scheduler:
            # print(optimizer.param_groups[0]['lr'], scheduler.get_lr())
            scheduler.step()

    return train_loss.item(), 100 * correct / total


def test(model, data_loader, criterion, device):
    model.eval()

    train_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss

            correct += (target == output.argmax(dim=1)).sum().item()
            total += target.size(0)

    return train_loss.item(), 100 * correct / total


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs_number, writer):
    accuracy_test_global = 0.0
    accuracy_train_global = 0.0

    for epoch in range(1, epochs_number):
        print("start epoch: {0:d}\n".format(epoch))
        loss_train, accuracy_train = train_epoch(model, train_loader, criterion, optimizer, device)
        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print("loss at epoch {0:d}: {1:f}, accuracy: {2:f}".format(epoch, loss_test, accuracy_test))

        writer.add_scalars(f'loss', {'train_metric': loss_train, 'test_metric': loss_test}, epoch)
        writer.add_scalars(f'accuracy', {'train_accuracy': accuracy_train, 'test_accuracy': accuracy_test}, epoch)

        accuracy_test_global = accuracy_test
        accuracy_train_global = accuracy_train

    return accuracy_test_global, accuracy_train_global


def train_warmup(model, train_loader, test_loader, criterion, device, writer, target_warmup_LR, iterations, start_warmupLR=0):
    # warmup lr from 0 to target LR (p.8)
    optimizer = torch.optim.SGD(model.parameters(), target_warmup_LR, momentum=0.9, weight_decay=1e-4)
    step = (target_warmup_LR - start_warmupLR) / (iterations * target_warmup_LR)
    batch_size = 128 # train_loader[0].shape[0]

    print('train_loader', len(train_loader))
    epochs_number = math.ceil(iterations / len(train_loader))

    lambda1 = lambda iter: step * iter
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    print('warmup start LR', scheduler.get_last_lr())
    '''accuracy_test_global, accuracy_train_global = train(
            model, train_loader, test_loader, criterion, optimizer, device, 10, writer, scheduler
    )'''
    accuracy_test_global = 0.0
    accuracy_train_global = 0.0

    print('epochs_number', epochs_number)

    for epoch in range(0, epochs_number):
        print(f'start warmup epoch: {epoch}/{epochs_number}')
        loss_train, accuracy_train = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print("loss at warmup epoch {0:d}: {1:f}, accuracy: {2:f}".format(epoch, loss_test, accuracy_test))
        print('warmup current LR:', scheduler.get_last_lr()[0])
        writer.add_scalars('loss warmup', {'train_metric': loss_train, 'test_metric': loss_test}, epoch)
        writer.add_scalars('accuracy warmup', {'train_accuracy': accuracy_train, 'test_accuracy': accuracy_test}, epoch)

        accuracy_test_global = accuracy_test
        accuracy_train_global = accuracy_train


    print('warmup end LR', scheduler.get_last_lr())

    return accuracy_test_global, accuracy_train_global

def train_iterations(model, start_LR, train_loader, test_loader, criterion, device, writer, iterations, round):
    '''
           We train the network for 30,000 iterations with SGD with momentum (0.9),
           decreasing the learning rate by a factor of 10 at 20,000 and 25,000 iterations. (p.8)
    '''
    # warmup lr from 0 to target LR (p.8)
    optimizer = torch.optim.SGD(model.parameters(), start_LR, momentum=0.9, weight_decay=1e-4)

    batch_size = 128 # train_loader[0].shape[0]

    epochs_number = math.ceil(iterations / len(train_loader))

    # lambda1 = lambda iter: 0.1 if iter > 2 * 10**4
    def lr_schedule(iter):
        if iter < 2 * 10 ** 4:
            return 1
        elif iter >= 2 * 10 ** 4 and iter < 25 * 10 ** 3:
            return 0.1
        else:
            return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    '''accuracy_test_global, accuracy_train_global = train(
            model, train_loader, test_loader, criterion, optimizer, device, 10, writer, scheduler
    )'''
    accuracy_test_global = 0.0
    accuracy_train_global = 0.0

    for epoch in range(1, epochs_number):
        print(f'start prune epoch: {epoch}/{epochs_number}')
        loss_train, accuracy_train = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print(f'loss at prune epoch {epoch} round {round}: {loss_test} accuracy: {accuracy_test}')
        print(f'prune current LR: {scheduler.get_last_lr()[0]}')
        writer.add_scalars(f'loss prune round-{round}', {'train_metric': loss_train, 'test_metric': loss_test}, epoch)
        writer.add_scalars(f'accuracy prune round-{round}', {'train_accuracy': accuracy_train, 'test_accuracy': accuracy_test}, epoch)

        accuracy_test_global = accuracy_test
        accuracy_train_global = accuracy_train

    return accuracy_test_global, accuracy_train_global

