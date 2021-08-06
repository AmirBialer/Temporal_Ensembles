# Temporal Ensembling for Semi-Supervised Learning
In this project we reproduce the results from [Temporal Ensembling for Semi-Supervised Learning](https://arxiv.org/abs/1610.02242) for [Pi model with augmentation](https://github.com/AmirBialer/FastSwa) on Cifar 10 dataset.
We then continue and suggest an improvement for Pi Model as we create an ensemble of it with xgboost.
Lastly, we performed Friedman test to check whether out improved model is significantly different than the other models: \{xgboost, pi model, improved pi model \}
And we continued with post hoc Nemenyi test and got our improved model is significantly better than: xgboost, pi model when the label ratio in the dataset is 0.1 or 0.05 for 20 different datasets.




## Credits
This Repo relies on the work of [fastswa](https://github.com/benathi/fastswa-semi-sup), [freitas](https://github.com/tensorfreitas/Temporal-Ensembling-for-Semi-Supervised-Learning), and [semi-supervised-learning] (https://github.com/siit-vtt/semi-supervised-learning-pytorch).

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

