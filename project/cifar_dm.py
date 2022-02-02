 
import pytorch_lightning as pl
from sympy import pretty, pretty_print
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, FashionMNIST
from shopee_dataset import ShopeeDataset
from torchvision import transforms
from typing import Optional
# from . import pretty_print
import pandas as pd
import os
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

        data = pd.read_csv('dataset/folds.csv')
        data['filepath'] = data['image'].progress_apply(lambda x: os.path.join('dataset','train_images', x))
        self.valid = data[data['fold']==0].reset_index(drop=True)
        self.train = data[data['fold']==1].reset_index(drop=True)
        # make training dataset here becasue this is needed in the pl_lightning module validation loop
        self.train_shopee = ShopeeDataset(
                csv=self.train,
                transforms=self.transform,
            )
        # make training dataset here becasue this is needed in the pl_lightning module validation loop
        self.test_shopee = ShopeeDataset(
                csv=self.valid,
                transforms=self.transform,
            )
        
        
    def prepare_data(self):
        # download
        # CIFAR10(self.data_dir, train=True, download=True)
        # CIFAR10(self.data_dir, train=False, download=True)
        # use only the CelebA dataset valid split for testing
        # CelebA(self.data_dir,  download=True, split = 'test', target_type = 'identity')
        # ==========================================================================
        #                             shopee dataset is below                                  
        # ==========================================================================
        pass
        

    def setup(self, stage: Optional[str] = None):
        '''
        called on every gpu separatey. Assign states like self.* here.

        Parameters
        ----------
        stage : Optional[str], optional
            fit --> MNIST train split used 
            test --> Shopee train split dataset used
            validate --> Shopee train split dataset used
        '''        

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # mnist_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            # shopee_full = self.shopee
            print(f"\n\n dataset size --> {len(self.train_shopee)} \n\n")
            # self.mnist_train, self.mnist_val = random_split(mnist_full, [45000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape) # X, Y ===> X.shape

        # Assign test dataset for use in test step
        if stage == "test" or stage is None:
            # self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            # self.caltech_valid = CelebA(self.data_dir,  download=True, split = 'test', target_type = 'identity', transform=self.transform)

            # self.shopee = ShopeeDataset(
            #     csv=self.valid,
            #     transforms=self.transform,
            # )
            print(f"\n\n dataset size --> {len(self.test_shopee)} \n\n")
        
   
        
        # Assign dataset to use for validation_step
        if stage == "validate" or stage is None:
            
            # self.shopee = ShopeeDataset(
            #     csv=self.valid,
            #     transforms=self.transform,
            # )
            print(f"\n\n dataset size --> {len(self.test_shopee)} \n\n")
    
    
    def train_dataloader(self):
        return DataLoader(self.train_shopee, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, num_workers=4)
        return DataLoader(self.test_shopee, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)
        return DataLoader(self.test_shopee, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='./dataset')
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--img_sz', type=int, default=224, help='size of image')
        parser.add_argument('--resize', type=int, default=250, help='resize the image to this size after which center crop is performed @ --img_sz flag')
        return parser



# if __name__ == '__main__':

# #     pretty_print(12)
#     # FACE alignment
#     # https://intellica-ai.medium.com/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7

#     dm = CIFAR_DataModule("dataset", img_sz=224, resize=250, batch_size=8)
#     dm.prepare_data()
#     dm.setup()
#     val_dl = enumerate(dm.test_dataloader())
#     trrain_dl = enumerate(dm.train_dataloader())
    
#     for k,v in val_dl:
#         print(f"\n\n val MINIBATCH --> {k} \n\n")
#         x,y = v
#         print(x.shape)
#         print(y.shape)
#         break
#     for k,v in trrain_dl:
#         print(f"\n\n train MINIBATCH --> {k} \n\n")
#         x,y = v
#         print(x.shape)
#         print(y.shape)
#         break