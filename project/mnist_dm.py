import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional
from pathlib import Path
# ========================================================================
#                             timm imports                                  
# ========================================================================
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm


model = timm.create_model('mnasnet_100', pretrained=True)
model.eval()


config = resolve_data_config({}, model=model)
transform = create_transform(**config)


# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
# img = Image.open(filename).convert('RGB')

# tensor = transform(img).unsqueeze(0) # transform and add batch dimension
# print(tensor.shape)


# class MNISTDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir: str):
#         super().__init__()
#         self.data_dir = data_dir
#         self.transform = transforms.Compose([transforms.Resize(self.SZ, self.SZ), transforms.ColorJitter(0.1, 0.1,0.1, 0.1) ,transforms.ToTensor(), transforms.Normalize(self.)])
#         # create an empty dir. if not exists
#         Path(self.data_dir).mkdir(parents=True, exist_ok=True)
#         # Setting default dims here because we know them.
#         # Could optionally be assigned dynamically in dm.setup()
#         self.dims = (1, 28, 28)

#     def prepare_data(self):
#         # download
#         MNIST(self.data_dir, train=True, download=True)
#         MNIST(self.data_dir, train=False, download=True)

#     def setup(self, stage: Optional[str] = None):

#         # Assign train/val datasets for use in dataloaders
#         if stage == "fit" or stage is None:
#             mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
#             self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

#             # Optionally...
#             self.dims = tuple(self.mnist_train[0][0].shape) # X, Y ===> X.shape

#         # Assign test dataset for use in dataloader(s)
#         if stage == "test" or stage is None:
#             self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

#             # Optionally...
#             # self.dims = tuple(self.mnist_test[0][0].shape)

#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=32)

#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=32)

#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=32)

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--data_dir', type=str, default='./dataset')
#         parser.add_argument('--learning_rate', type=float, default=0.0001)
#         return parser