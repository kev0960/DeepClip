import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def main():
    model = models.resnet152(pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    traindir = "/"

