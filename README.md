# The code for ICASSP paper *Ultrasound Image Classification Improved by Local Texture Quantisation*

The code covers the experiments in the paper i.e. the Thyroid nodules classification (DDTI dataset), and the Breast nodules classification on (BUSI dataset).

## To run the code

0. The code is based on Pytorch 1.7.1. It uses a 3rd-party lib [einops](https://github.com/arogozhnikov/einops).

1. To run the two paper experiments, just run the file **Exp2-LTQ.py**. The file defines the training and testing process. At the end of the file, the file calls **loop($\cdot$)** function twice, starting the two experiments.

## To run the simple baselines (e.g. ResNet, DenssNet)

Just run the **Exp1-Baseline-models.py**. The **loop($\cdot$)** function at the end of the file accepts the baseline model name (*model_name* argument). The model name should be a string, which is one of the predefined model names by PyTorch in `torchvision.utils`.

## The current experiments results:

### 1. DDTI dataset

```
# code to run the experiment
loop(f'Exp2-PAPER-c-{ticks}', **{
    'use_pretrained_ae' : None, 
    'input_channels' : ['image'],
    'n_epoch1': 4000, 
    'n_epoch2' : 200, 
    'Yu' : True, 
    'data' : 'DDTI',
    'n_embed' : 64,
    'ae_plan' : [1, 32, 64, 128, 2048],
    'cl_plan' : [64, 64, 64],
    'centre_crop' : False,
}) 
```
Result:
```
              precision    recall  f1-score   support

           0     0.9040    0.9286    0.9161       294
           1     0.9874    0.9827    0.9851      1680

    accuracy                         0.9747      1974
   macro avg     0.9457    0.9557    0.9506      1974
weighted avg     0.9750    0.9747    0.9748      1974
```
Confusion matrix:
```
 [[ 273   21]
  [  29 1651]]
```

### 2. BUSI dataset 

```
# code to run the experiment
loop(f'Exp2-PAPER-c-{ticks}', **{
    'use_pretrained_ae' : None,
    'input_channels' : ['mask'],
    'n_epoch1': 3000, 
    'n_epoch2' : 200, 
    'Yu' : True, 
    'data' : 'BUSI',
    'n_embed' : 32,
    'ae_plan' : [1, 32, 64, 256],
    'cl_plan' : [64, 64, 64, 64]
}) 
```
Result:
```
              precision    recall  f1-score   support

           0     0.9337    0.9851    0.9587      1743
           1     0.9701    0.8737    0.9194       966

    accuracy                         0.9454      2709
   macro avg     0.9519    0.9294    0.9390      2709
weighted avg     0.9467    0.9454    0.9447      2709
```
Confusion matrix:
```
 [[1717   26]
  [ 122  844]]
```