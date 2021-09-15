import torch
import torch.nn as nn
import torch.nn.functional as F
from prune_utils import prune_model

import torch.optim as optim
from sklearn.metrics import accuracy_score

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, inside_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # really stride doesn't depend on it
        # self.stride = int(inside_channels / in_channels)
        # relu before addition
        # There are another residual block types: 
        # e.g. full-pre-activation BN-> ReLU->conv->BN-> ReLU->conv
        # see https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        self.stride = stride
        
        # actually you don't need ReLU as field here, moreover 2
        self.conv1 = nn.Conv2d(in_channels, inside_channels, 3, padding=1, stride=self.stride, bias=False)
        self.bn1 = nn.BatchNorm2d(inside_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inside_channels, inside_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inside_channels)
        self.relu2 = nn.ReLU()
        
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, inside_channels, 1, stride=self.stride, bias=False),
                nn.BatchNorm2d(inside_channels)
            )
            
            torch.nn.init.xavier_uniform_(self.downsample[0].weight)
            
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        # torch.nn.init.xavier_uniform_(self.conv2.weight)
        
            
        
    def forward(self, x):
        inp = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        if self.stride != 1:
            inp = self.downsample(inp)
        
        return out + inp
    
    
class CifarResNet20(nn.Module):
    # by https://medium.com/@chenchoulo/build-up-and-train-a-resnet-20-model-on-caffe2-98b3f31662c
    def __init__(self, num_classes=10):
        super(CifarResNet20, self).__init__()
        # input_channels, output_channels, kernel
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, stride=1, bias=False)
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
    
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2_1 = BasicBlock(16, 16)
        self.conv2_2 = BasicBlock(16, 16)
        self.conv2_3 = BasicBlock(16, 16)
        
        self.conv3_1 = BasicBlock(16, 32, stride=2)
        self.conv3_2 = BasicBlock(32, 32)
        self.conv3_3 = BasicBlock(32, 32)
        
        self.conv4_1 = BasicBlock(32, 64, stride=2)
        self.conv4_2 = BasicBlock(64, 64)
        self.conv4_3 = BasicBlock(64, 64)
        
        self.pool5 = nn.AvgPool2d(8)
        self.fc5 = nn.Linear(64, num_classes)
        
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.fc5(x)
        
        return x

    @property
    def weights(self):
        flag = True
        for k in self.named_parameters():
            if '_orig' in k:
                flag=False

        if flag:
            print('Force identity pruning')
            prune_model(self)

        # self.named_
        return dict(self.named_parameters())
        # return {k:i for k,i in self.state_dict().items() if '_mask' not in k }

    @property
    def mask(self):
        return {k:i for k,i in self.state_dict().items() if '_mask' in k}

    def load_weights_and_mask(self, weights, mask):
        import copy
        # prune.identity
        # prune.custom_from_mask
        # prev = copy.deepcopy(self.state_dict()['conv3_2.conv2.weight_orig'])
        sum_ = 0.0
        for k, i in weights.items():
            # print(f'diff: {k:30s} {torch.sum(self.state_dict()[k] - i)}')
            sum_ += torch.abs(torch.sum(self.state_dict()[k] - i))

        print(f'total diff sum {sum_}')

        self.load_state_dict(weights, strict=False)
        self.load_state_dict(mask, strict=False)
