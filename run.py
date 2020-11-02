import os, torchvision, torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from dataset import H5Dataset
from dataset.input import InputTransform, S1C1Transform
from dataset.filters import DoGKernel, GaborKernel, Filter

from models import MozafariMNIST, KheradpishehMNIST
from estimator.supervised import SupervisedEstimator
from estimator.unsupervised import UnsupervisedEstimator
from utils import read_yaml, write_perf_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.isfile('envs.yaml'):
    envs = read_yaml('envs.yaml')
else:
    envs = {'data_path': './data/'}

kernels = [DoGKernel(3,3/9,6/9),
		DoGKernel(3,6/9,3/9),
		DoGKernel(7,7/9,14/9),
		DoGKernel(7,14/9,7/9),
		DoGKernel(13,13/9,26/9),
		DoGKernel(13,26/9,13/9)]

filter = Filter(kernels, padding = 6, thresholds = 50)
s1c1 = S1C1Transform(filter)

data_root = "data"
batch_size = 128
number_of_class = 2
# train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1)
# test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1)
# train_dataset = random_split(train_dataset, [10000, len(train_dataset)-10000])[0]
# test_dataset = random_split(test_dataset, [10000, len(test_dataset)-10000])[0]
# train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True, num_workers=4, shuffle=False)

## BRATS ##
train_dataset = H5Dataset(f_path=envs['training_data_path'], length=None,transform=s1c1)
test_dataset = H5Dataset(f_path=envs['testing_data_path'], length=None,transform=s1c1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

data_len = len(train_dataset)
print(data_len)
print('\033[94m'+'\nStarting UNSUPERVISED training :\n\033[0m')

epochs = [5, 10]
net = KheradpishehMNIST().to(device)
estimator = UnsupervisedEstimator(net, save_path='./saved_models/')
train_perf, test_perf = estimator.eval(train_dataloader=train_dataloader, 
									test_dataloader=test_dataloader,
									epochs=epochs)

with open('performance.txt', 'a') as f:
	f.write('\n######## START ########')
write_perf_file(net, data_len, estimator.type, kernels, epochs, envs['training_data_path'], train_perf, test_perf)

print('\033[94m'+'\nStarting SUPERVISED training :\n\033[0m')

epochs = [2, 4, 10]
net = MozafariMNIST(input_channels=6, output_channels=40, num_classes=number_of_class).to(device)
estimator = SupervisedEstimator(net, save_path='./saved_models/')
train_perf, test_perf = estimator.eval(train_dataloader=train_dataloader, 
									test_dataloader=test_dataloader,
									epochs=epochs)
write_perf_file(net, data_len, estimator.type, kernels, epochs, envs['training_data_path'], train_perf, test_perf)

with open('performance.txt', 'a') as f:
	f.write('\n######## END ########')