from torchvision import datasets, transforms
from my_resnet import CifarResNet20
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pytorch_model_summary import summary
from prune_utils import print_statisitcs_by_masks, prune_model
from train import train_warmup, train_iterations, test
import argparse
import sys, os
import datetime
import copy

def load_datasets(root='./data', drop_last=True, batch_size_train=512, batch_size_test=512):
    batch_size = 512

    cifar10_train = datasets.CIFAR10(root=root, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))

    train_loader = torch.utils.data.DataLoader(
            cifar10_train,
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            drop_last=drop_last
    )

    cifar10_test = datasets.CIFAR10(root=root, train=False, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           ]))

    test_loader = torch.utils.data.DataLoader(
            cifar10_test,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
    )

    return train_loader, test_loader





def main(prune_percent, rounds, start_LR=0.01, warmup_iterations=10 ** 4, prune_iterations=3 * 10 ** 4, warmup_load=None):
    # according to Frankle-Corbin: p.3 iterations/batch 30k/128
    train_loader, test_loader = load_datasets(batch_size_train=128)

    model = CifarResNet20()
    print(summary(model, torch.zeros((1, 3, 32, 32)), show_input=True))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)
    # tensorboard settings

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'logs/{current_time}')
    # mkdir for models
    path_to_model_dir = f'models/{current_time}'
    os.mkdir(path_to_model_dir)

    # save original weights
    original_weights = copy.deepcopy(model.weights)

    # warmup
    if warmup_load:
        model.load_state_dict(torch.load(warmup_load))
    else:
        if warmup_iterations:
            train_warmup(model, train_loader, test_loader, criterion, device, writer, start_LR, warmup_iterations)
            torch.save(model.state_dict(), f'{path_to_model_dir}/model-warmup-{start_LR}.model')

    warmup_weights = model.weights

    accuracy_test_global, accuracy_train_global = 0, 0

    for round in range(rounds):
        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print(f'After training: sparsity level {1 - (1 - prune_percent) ** round} loss_test {loss_test}, accuracy_test {accuracy_test} ')
        writer.add_scalars('Common/test_loss', {'loss_test' : loss_test, 'sparsity %': 1 - (1 - prune_percent) ** round}, round)
        writer.add_scalars('Common/test_accuracy', {'accuracy_test': accuracy_test, 'sparsity %': 1 - (1 - prune_percent) ** round}, round)
        # add early stopping and save the best
        print('#' * 100)
        print(f'Pruning...')
        prune_model(model, prune_percent)
        mask = model.mask

        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print(f'After pruning: loss_test {loss_test}, accuracy_test {accuracy_test}')
        # return to original weights
        print(f'Load original weights...')
        model.load_weights_and_mask(original_weights, mask)

        loss_test, accuracy_test = test(model, test_loader, criterion, device)
        print(f'After loading original: loss_test {loss_test}, accuracy_test {accuracy_test}')
        # train again
        print(f'Pruning round number {round} / {rounds}')
        accuracy_test_global, accuracy_train_global = train_iterations(
            model, start_LR, train_loader, test_loader, criterion, device, writer, iterations=prune_iterations, round=round
        )

        torch.save(model.state_dict(),
                   f'{path_to_model_dir}/model-{round}-from-{rounds}-LR-{start_LR}-{accuracy_test_global}-{accuracy_train_global}.model')
        '''print(f'Pruning...')
        prune_model(model, prune_percent)
        mask = model.mask
        # return to original weights
        print(f'Load original weights...')
        model.load_weights_and_mask(original_weights, mask)'''


    print('save final model...')
    torch.save(model.state_dict(),
               f'{path_to_model_dir}/model-{rounds}--LR-{start_LR}{accuracy_test_global}-{accuracy_train_global}-final.model')
    writer.close()


if __name__ == '__main__':
    # resnet-18 experiment settings (see p.40)


    parser = argparse.ArgumentParser(description='ResNet lottery tickets reproduction')
    parser.add_argument('--start_lr', type=float, default=0.01)
    parser.add_argument('--pruning_rate', type=float, default=0.2)
    parser.add_argument('--pruning_rounds', type=int, default=16)

    parser.add_argument('--warmup_iterations', type=int, default=10000)
    parser.add_argument('--prune_iterations', type=int, default=30000)
    parser.add_argument('--save_warmup')
    parser.add_argument('--load_warmup', type=str, default='model-warmup-10000.model')
    parser.add_argument('--save_pune_round', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    main(args.pruning_rate, args.pruning_rounds, args.start_lr, args.warmup_iterations, args.prune_iterations)
    # main(0.1, 10)