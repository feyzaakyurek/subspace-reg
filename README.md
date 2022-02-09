# Subspace Regularizers

This repo contains the source code for the following paper:  

[**"Subspace Regularizers for Few-Shot Class Incremental Learning"**](https://arxiv.org/abs/2110.07059), Afra Feyza Akyürek, Ekin Akyürek, Derry Wijaya, Jacob Andreas. ICLR 2022.


## Installation

Setup the conda environment using `environment.yml`.

```
git clone https://github.com/feyzaakyurek/subspace-reg.git
cd subspace-reg
conda env create -f environment.yml
```
and activate it.

## Download Data and Pretrained Models

### Multi Session

If running `multi-session` please download data from this [link](https://drive.google.com/file/d/14aOw3G3uOaaq7jPswLsDE080K0K-tkwJ/view?usp=sharing), uncompress the contents under `data/miniImageNet` and place `data` under this repo. You should have two files under `data/miniImageNet`: `all.pickle` and `class_names.txt`.

Download pretrained models for `multi-session` from [here](https://drive.google.com/file/d/1VzjnZcjgwlQe7CK-Sl730In-WNPvom0b/view?usp=sharing) and place them under `dumped/backbones/continual/resnet18`. The numbered folder names refer to the seeds. Eventually you will have `dumped/backbones/continual/resnet18/1`, `dumped/backbones/continual/resnet18/2`, etc.

### Single Session
If running `single-session`, download data from [here](https://github.com/renmengye/inc-few-shot-attractor-public) and place it under `data/[DATASETNAME]` replacing DATASETNAME with miniImageNet or tieredImageNet.

Download pretrained model TBD.

## Running

Sample scripts are under `scripts/continual` for `multi-session` along with the scripts for `single-session`. In every script you will see a nested for loop which we used for hyperparameter tuning. Each experiment is run for 10 different seeds. If you are interested in only a single run, scroll down to the bottom for the respective command in every `.sh` file. 

The existing commands use memory (+M setup in the paper). In order to turn off memory replay, simply remove the options `--n_base_support_samples 1` and `--memory_replay 1` which will set them to zero. Make sure to edit your `EXP_FOLDER` in the scripts to avoid overriding your previous experiments.

## Contacts
Feel free to reach out to with questions.

Afra Feyza Akyürek (akyurek@bu.edu)  
Ekin Akyürek (akyurek@mit.edu)

## Acknowledgment
A significant portion of this repository is based on [RFS source code](https://github.com/WangYueFt/rfs).

```
@article{akyurek2021subspace,
  title={Subspace Regularizers for Few-Shot Class Incremental Learning},
  author={Aky{\"u}rek, Afra Feyza and Aky{\"u}rek, Ekin and Wijaya, Derry and Andreas, Jacob},
  journal={arXiv preprint arXiv:2110.07059},
  year={2021}
}
```
