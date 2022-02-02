from cv2 import transform
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
tqdm.pandas()
import torch
from torchvision import transforms

class ShopeeDataset(Dataset):
    def __init__(self, csv, transforms=None):
        '''
        initilizer for shopee dataset class

        Parameters
        ----------
        csv : str
            path of the csv-kfolds csv file
        transforms : transforms.Compose, optional
            must be defined using transforms.Compose  NOT using albumentations
        '''        

        self.csv = csv.reset_index()
        self.augmentations = transforms
        

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index].tolist()
        # Console().print(row)
        
        text = row[4]
        
        assert os.path.exists(row[7]) , "image does not exist"
        image = Image.open(row[7]).convert('RGB')

        if self.augmentations:
            image = self.augmentations(image)
        else:
            # if no augmentation is applied then convert the image into the [C, H, W] format
            trf = transforms.Compose([transforms.ToTensor()])
            image = trf(image).type(torch.float32)
        
        return image,torch.tensor(row[5]).type(torch.long)


# ==========================================================================
#                             how to use K-fold split                                  
# ==========================================================================

# >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# >>> y = np.array([0, 0, 1, 1])
# >>> skf = StratifiedKFold(n_splits=2)
# >>> skf.get_n_splits(X, y)
# 2
# >>> print(skf)
# StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
# >>> for train_index, test_index in skf.split(X, y):
# ...     print("TRAIN:", train_index, "TEST:", test_index)
# ...     X_train, X_test = X[train_index], X[test_index]
# ...     y_train, y_test = y[train_index], y[test_index]


    



# for k,v in valid_dataset:
#     print(k.shape,v.shape)
#     break