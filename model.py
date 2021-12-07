#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from architectures.wide_resnet import *
from torch.backends import cudnn
from utils import *

cudnn.benchmark = True

print(f"Number of GPUs = {torch.cuda.device_count()}")
valid_size = 1024
batch_size = 256

'''Basic neural network architecture (from pytorch doc).'''
# torch.distributed.init_process_group(
#     backend="nccl",
#     init_method=get_init_file().as_uri(),
#     world_size=1,
#     rank=0,
# )


class Net(nn.Module):
    model_file = "model_copy.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        print(self.model_file)
        self.lbda = [0.51101563, 0.48898437]

        self.models = [NormalizedModel(WideResNet28x10(num_classes=10),
                                       mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])] * 2



        # for i in range(2):
        #     self.models[i].cuda(0)
        #     self.models[i] = torch.nn.parallel.DistributedDataParallel(self.models[i], device_ids=[0],
        #                                                                output_device=1)

        self.models = nn.ModuleList(self.models)
        self.mix_model = MixedModel(self.models, self.lbda)

    def forward(self, x):
        return self.mix_model(x)

    # def save(self, model_file):
    #     '''Helper function, use it to save the model weights after training.'''
    #     torch.save(self.state_dict(), model_file)

    def load(self):
        state_dict = torch.load(self.model_file)
        self.models.load_state_dict(state_dict)
        self.mix_model = MixedModel(self.models, self.lbda)

    def load_for_testing(self, project_dir=None):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.

           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''
        self.load()

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].cuda(), data[1].cuda()
            # calculate outputs by running images through the network

            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total




def get_train_loader(dataset, valid_size=1024, batch_size=256):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train


def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid


def main():
    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = net.cuda()

    test_transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=test_transform)
    valid_loader = get_validation_loader(cifar, valid_size)

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.

    net.module.load()

    acc = test_natural(net, valid_loader)

    print("Model natural accuracy (valid): {}".format(acc))


if __name__ == "__main__":
    main()
