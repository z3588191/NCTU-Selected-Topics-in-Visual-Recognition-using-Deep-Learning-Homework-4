# NCTU Selected Topics in Visual Recognition using Deep Learning, Homework 4
Code for Single Image Super Resolution.


## Hardware
The following specs were used to create the submited solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- NVIDIA GeForce 2080Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#Installation)
2. [Dataset Download](#Dataset-Download)
3. [Prepapre Dataset](#Prepare-Dataset)
4. [Train models](#Train-models)
5. [Evaluation](#Evaluation)
6. [Reference](#Reference)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hw4 python=3.7
source activate hw4
pip install -r requirements.txt
```

## Dataset Download
Dataset download link is in Data section of [HW4](https://drive.google.com/drive/u/3/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x)

## Prepare Dataset
After downloading, the data directory is structured as:
```
${ROOT}
  +- testing_lr_images
  |  +- 00.png
  |  +- 01.png
  |  +- ...
  +- training_hr_images
  |  +- 2092.png
  |  +- 8049.png
  |  +- ...
```

### Train models
To train models, run following commands.
```
$ python train.py 
```


## Evaluation
After training, you can get model weight `RFDN.pkl`:
```
${ROOT}
  +- RFDN.pkl
  +- eval.py
  +- ...
```

To get SR image of testing LR image, run following commands.
```
$ python eval.py 
```

## Reference
[torchvision](https://github.com/pytorch/vision)

[RFDN](https://github.com/njulj/RFDN)
