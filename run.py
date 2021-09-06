import os, torchvision, torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split

from dataset import H5Dataset
from dataset.input import InputTransform, S1C1Transform
from dataset.filters import DoGKernel, GaborKernel, Filter

from models import MozafariMNIST, KheradpishehMNIST
from estimator.supervised import SupervisedEstimator
from estimator.unsupervised import UnsupervisedEstimator
from utils import write_perf_file


task = 'mnist'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kernels = [DoGKernel(3,3/9,6/9),
	DoGKernel(3,6/9,3/9),
	DoGKernel(7,7/9,14/9),
	DoGKernel(7,14/9,7/9),
	DoGKernel(13,13/9,26/9),
	DoGKernel(13,26/9,13/9)]

filter = Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1C1Transform(filter)

data_root = "./data"
number_of_class = 10
train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1)
test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1)
train_dataset = random_split(train_dataset, [10000, len(train_dataset)-10000])[0]
test_dataset = random_split(test_dataset, [10000, len(test_dataset)-10000])[0]

train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True, num_workers=4, shuffle=False)
data_len = len(train_dataset)

print('\033[94m'+'\nStarting SUPERVISED training :\n\033[0m')

epochs = [1, 1, 1]
net = MozafariMNIST(input_channels=6, features_per_class=10, num_classes=number_of_class).to(device)
estimator = SupervisedEstimator(net, save_path='./saved_models/')
train_perf, test_perf = estimator.eval(train_dataloader=train_dataloader, 
									test_dataloader=test_dataloader,
									epochs=epochs)
write_perf_file(net, task, data_len, estimator.type, kernels, epochs, train_perf, test_perf)