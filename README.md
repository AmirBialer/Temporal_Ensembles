# Temporal Ensembling for Semi-Supervised Learning


This Repo is heavily relying on the work of [fastswa](https://github.com/benathi/fastswa-semi-sup), [freitas](https://github.com/tensorfreitas/Temporal-Ensembling-for-Semi-Supervised-Learning), and [semi-supervised-learning] (https://github.com/siit-vtt/semi-supervised-learning-pytorch).
We reproduce the results from [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242) for Pi model with augmentation.

## Preparing Packages and Data

The code runs on Python 3 with Pytorch 0.3. The following packages are also required.
```
pip install scipy tqdm matplotlib pandas msgpack
```
moreover, this code relies on the mentioned above packages at their specific versions, so its best to use the requirements.txt file.



## Pi_Model training and evaluation


```
python train.py --path "path_to_datasets_folder" --improved_model True/False
```

