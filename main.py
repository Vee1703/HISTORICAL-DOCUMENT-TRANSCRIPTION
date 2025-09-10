import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, USPS

transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
from trainer import Trainer

source_train_dataset = MNIST(root='/DATA/diwan1/data', train=True, download=True,transform=transform)
target_train_dataset = USPS(root='/DATA/diwan1/data', train=True, download=True,transform=transform)
source_test_dataset = MNIST(root='/DATA/diwan1/data', train=False, download=True,transform=transform)
target_test_dataset = USPS(root='/DATA/diwan1/data', train=False, download=True,transform=transform)

source_train_loader = DataLoader(source_train_dataset, batch_size=16, shuffle=True)
target_train_loader = DataLoader(target_train_dataset, batch_size=16, shuffle=True) 
source_test_loader = DataLoader(source_test_dataset, batch_size=16, shuffle=True) 
target_test_loader = DataLoader(target_test_dataset, batch_size=16, shuffle=True)

trainer = Trainer(source_train_loader, target_train_loader, source_test_loader, target_test_loader)
trainer.train()
