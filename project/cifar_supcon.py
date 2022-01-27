# supcon loss is a supervised contrastive learning loss. i.e. it needs the labels to perform learning as compared to SimSiam and SimCLR.

#%%
from argparse import ArgumentParser
from torch.optim.lr_scheduler import MultiStepLR
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
# from torch.utils.data import DataLoader, random_split
from typing import List
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
import timm
from cifar_dm import CIFAR_DataModule
from pytorch_metric_learning.losses import SupConLoss
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

# wandb.login()

class LitClassifier(pl.LightningModule):
    def __init__(self, embed_sz : int , 
                        steps : List, 
                        warmup_epochs : int, 
                        lr : float = 1e-3,  
                        gamma : float = 0.1, **kwargs):
        super().__init__()
        self.embed_sz = embed_sz
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.gamma = gamma
        self.steps = [int(k) for k in steps]
        # define the backbone network
        self.backbone = timm.create_model('mnasnet_100', pretrained=True)
        # put backbone in train mode
        self.backbone.train()
        in_features = self.backbone.classifier.in_features
        self.project = torch.nn.Linear(in_features, self.embed_sz)
        
        self.backbone.classifier = torch.nn.Identity()        

        self.supcon_head = SupConLoss(temperature=0.1)
        # self.activation = torch.nn.LeakyReLU(negative_slope=0.1)


        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x)
        x = self.project(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(x.shape)
        # print(y.shape)
        embeddings = self(x)
        loss = self.supcon_head(embeddings, y)
        # for logging to the loggers
        self.log("train_loss", loss) 

        return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss = self.supcon_head(embeds, y)
        # for logging to the loggers
        self.log('val_loss', loss)
        return {"loss":loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeds = self(x)
        loss = self.supcon_head(embeds, y)
        self.log('test_loss', loss)
        return {"test_loss":loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # sched = MultiStepLR(opt, milestones=self.hparams.steps, gamma=self.hparams.gamma)
        
        

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, self.hparams.max_epochs, eta_min=0
        # )
        lr_scheduler = LinearWarmupCosineAnnealingLR(opt, self.warmup_epochs, self.trainer.max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)
        return [opt], [lr_scheduler]

        # return {"optimizer": opt, 
        #         "lr_scheduler": 
        #                     {
        #                             # REQUIRED: The scheduler instance
        #                             "scheduler": sched,
        #                             # The unit of the scheduler's step size, could also be 'step'.
        #                             # 'epoch' updates the scheduler on epoch end whereas 'step'
        #                             # updates it after a optimizer update.
        #                             "interval": "epoch",
        #                             # How many epochs/steps should pass between calls to
        #                             # `scheduler.step()`. 1 corresponds to updating the learning
        #                             # rate after every epoch/step.
        #                             "frequency": 1,
                                    
        #                     }
        # }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--embed_sz', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--steps', nargs='+', required=True)
        parser.add_argument('--warmup_epochs', type=int, default=1)
        return parser


def cli_main():
    pl.seed_everything(1000)
    wandb.init()
    wandb.run.log_code(".") # all python files uploaded
    wandb.login()

    
    checkpoint_callback = ModelCheckpoint(filename="checkpoints/cifar10-{epoch:02d}-{val_loss:.6f}", monitor='val_loss', mode='min', )
    lr_callback = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(project='CIFAR-10', # group runs in "MNIST" project
                           log_model='all', # log all new checkpoints during training
                            name="supcon loss")
    # ------------          
    # args
    # ------------
    parser = ArgumentParser()
    
    
    #  trainer CLI args added
    parser = pl.Trainer.add_argparse_args(parser)
    # model specific args
    parser = LitClassifier.add_model_specific_args(parser)
    # dataset specific args
    parser = CIFAR_DataModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # ------------
    # data
    # ------------
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    # val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)
    dm = CIFAR_DataModule(**vars(args)) # vars converts Namespace --> dict, ** converts to kwargs
    # ------------
    # model
    # ------------
    model = LitClassifier(**vars(args))
    wandb_logger.watch(model)
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(lr_callback)
    trainer.logger = wandb_logger
    trainer.tune(model, dm)

    # log args to wandb
    args.batch_size = model.hparams.get('batch_size')
    # dm.hparams.batch_size = args.batch_size
    dm.hparams.batch_size = 512
    print(f"\n\n batch size -----> {args.batch_size}\n\n")
    wandb.config.update(vars(args))

    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test()


if __name__ == '__main__':
    cli_main()
