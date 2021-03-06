:dagger: :fire: :warning:  📒

auto_scale_batch size is not working.

### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/TalhaUsuf/SupCon_loss.git

# install project   
cd SupCon_loss
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, run the trainig with:   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python cifar_supcon.py --embed_sz 128 --gamma 0.1 --steps 3 4 --data_dir dataset --bs 8 --img_sz 20 --resize 25 --auto_scale_batch_size True     
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

# Distributed Training

:warning: :fire: :anger: :bangbang:

> auto scale batch size is inherently not supported for distributed training by **pytorch lightning**
----
:arrow_forward: 



For distributed training

```bash
python cifar_supcon.py --run_name arcface_2_optimizers_autoscale --warmup_epochs 4 --max_epochs 10 --auto_scale_batch_size False --embed_sz 128 --gamma 0.1 --steps 3 4 --gpus 2 --data_dir dataset --img_sz 224 --resize 250 --precision 16 --amp_backend native --num_nodes 1 --strategy ddp --batch_size 200
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
