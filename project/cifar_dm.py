 
import pytorch_lightning as pl
from sympy import pretty, pretty_print
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision import transforms
from typing import Optional
# from . import pretty_print
from PIL import Image
from pathlib import Path



class CIFAR_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, img_sz : int, resize : int, batch_size : int, *args, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.img_sz = img_sz
        self.resize = resize
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                                    transforms.Resize(size=self.resize, interpolation=Image.BICUBIC),
                                    transforms.CenterCrop(size=(self.img_sz, self.img_sz)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
                                    # transforms.Normalize(mean=[0.4850,], std=[0.2290,])
        ])
        # create an empty dir. if not exists
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        
        # try updating the lightning to see if it works
        self.save_hyperparameters()
        print("==================================")
        print(self.hparams)
        print("==================================")

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        # CIFAR10(self.data_dir, train=False, download=True)
        # use only the CelebA dataset valid split for testing
        # CelebA(self.data_dir,  download=True, split = 'test', target_type = 'identity')
        FashionMNIST(root = self.data_dir, train = False,  download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            print(f"\n\n dataset size --> {len(mnist_full)} \n\n")
            self.mnist_train, self.mnist_val = random_split(mnist_full, [45000, 5000])

            # Optionally...
            self.dims = tuple(self.mnist_train[0][0].shape) # X, Y ===> X.shape

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            # self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            # self.caltech_valid = CelebA(self.data_dir,  download=True, split = 'test', target_type = 'identity', transform=self.transform)
            self.lfw_test = FashionMNIST(root = self.data_dir, train = False,  download=True, transform=self.transform)
            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)
        
        # Assign dataset to use for validation_step
        if stage == "validate" or stage is None:
            # self.caltech_valid = CelebA(self.data_dir,  download=True, split = 'test', target_type = 'identity', transform=self.transform)
            self.lfw_test = FashionMNIST(root = self.data_dir, train = False,  download=True, transform=self.transform)
    
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, num_workers=4)
        return DataLoader(self.lfw_test, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)
        return DataLoader(self.lfw_test, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='./dataset')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--img_sz', type=int, default=224, help='size of image')
        parser.add_argument('--resize', type=int, default=250, help='resize the image to this size after which center crop is performed @ --img_sz flag')
        return parser



# if __name__ == '__main__':

#     pretty_print(12)
    # FACE alignment
    # https://intellica-ai.medium.com/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7

    # dm = CIFAR_DataModule("dataset", img_sz=224, resize=250, batch_size=8)
    # dm.prepare_data()
    # dm.setup()
    # train_dl = enumerate(dm.train_dataloader())

    # for k,v in train_dl:
    #     print(f"\n\n MINIBATCH --> {k} \n\n")
    #     x,y = v
    #     print(x.shape)
    #     print(y.shape)