- [Your Project Name](#your-project-name)
  - [Description](#description)
  - [How to run](#how-to-run)
  - [Base tester flag](#base-tester-flag)
  - [Imports](#imports)
    - [Citation](#citation)

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
 Next, run the DDP mode training using the following command:   
 ```bash
# module folder
cd project

# run module   
cifar_supcon.py --embed_sz 256 --gamma 0.1 --steps 3 4 --data_dir dataset --img_sz 224 --resize 250 --auto_scale_batch_size False --strategy ddp --gpus 2 --precision 16 --batch_size 250 --max_epochs 10 --warmup_epochs 2 --log_every_n_steps 5 --lr 0.0001
```
## Base tester flag

`get_accuracy` has an important flag docs [here](https://kevinmusgrave.github.io/pytorch-metric-learning/testers/) `embeddings_come_from_same_source` which is used to determine if the embeddings are from the same source.


Lone query labels 🔥 🗡️ :fork_and_knife:

If some query labels don't appear in the reference set, then it's impossible for those labels to have non-zero k-nn accuracy. Zero accuracy for these labels doesn't indicate anything about the quality of the embedding space. **So these lone query labels are excluded from k-nn based accuracy calculations**.

---
:hot_pepper: :volcano:  :waning_gibbous_moon:

For example, if the input query_labels is `[0,0,1,1]` and reference_labels is `[1,1,1,2,2]`, then 0 is considered a lone query label.

See more docs here: https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/

---
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

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
