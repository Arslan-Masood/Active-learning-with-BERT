#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import deepchem as dc
from deepchem.data import NumpyDataset
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')
import os
import torch

import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


import pandas as pd
import math 
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.MolStandardize import rdMolStandardize
IPythonConsole.drawOptions.comicMode=True
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
import rdkit
from rdkit.Chem.SaltRemover import SaltRemover
print(rdkit.__version__)
import numpy as np

# split into train and test based on scafold split
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import StratifiedGroupKFold

from torch.utils.data import DataLoader

import wandb
import os
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")

import torch.multiprocessing as mp

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import ParameterGrid
import gc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pytorch_lightning import seed_everything
from functools import partial


# In[2]:


from utils.data_utils import scafoldsplit_train_test, convert_to_dataframe, get_stratified_folds
from utils.data_utils import dataloader_for_numpy
from utils.utils import wandb_init_model, get_model_predictions_MT, compute_binary_classification_metrics_MT
from utils.models import Vanilla_MLP_classifier


# In[3]:


config = {
        # directories
        "project_name": "Tox21_BERT",
        "metadata_dir": '/projects/home/mmasood1/trained_model_predictions/Tox21/Frozen_BERT/baseline_with_BERT/Final_model/',
        "target_file": "/projects/home/mmasood1/arslan_data_repository/Tox21/complete_Tox21.csv",
        "BERT_features_file":"/projects/home/mmasood1/arslan_data_repository/Tox21/Tox21_BERT_features.csv",
        "model_weights_dir" : '/projects/home/mmasood1/Model_weights/Tox21/',
        "pos_weights": "/projects/home/mmasood1/arslan_data_repository/Tox21/pos_weights.csv",
        "class_weights": "/projects/home/mmasood1/arslan_data_repository/Tox21/target_weights.csv",
        
        # data
        "features_type" :"BERT",
        "FP_size" : 1024,
        "train_frac": 0.8,

        # architechture
        "input_dim": 768,
        "hidden_dim": 128,
        "depth" : 1,
        "BatchNorm1d": True,
        "use_skip_connection": True,
    
        # training
        "optim": 'Adam',#SGD
        "lr_schedulers": "CosineAnnealingLR",
        "lr": 1e-3,
        "l2_lambda": 0.0,
        "optm_l2_lambda": None,
        "epochs": 410,
        "compute_metric_after_n_epochs": 5,
        "batch_size": 64,
        "EarlyStopping": False, 
        "pretrained_model": False,

        # loss
        "missing" : 'nan',
        "alpha": 0,
        "beta": 0,
        "gamma":0,

        "gpu": [0],
        "accelerator": "gpu",
        "device" :torch.device("cuda"),
        "return_trainer": True, 
        "save_predicitons" : True,
        "Final_model": True
    }

# get targets information
data = pd.read_csv(config["target_file"])
target_names = data.loc[:, "NR-AR":"SR-p53"].columns.tolist()

config["num_of_tasks"] = len(target_names)
config["selected_tasks"] = target_names

config["seed"] = 42

if config["Final_model"]:
    fold_list = [0]
else:
    fold_list = [0,1,2,3,4]


# In[4]:


# Splitting by using deepchem
train_set, test_set = scafoldsplit_train_test(config)
train_set = convert_to_dataframe(train_set, config["selected_tasks"])
test_set = convert_to_dataframe(test_set, config["selected_tasks"])


# In[5]:


# Get MolBERT features
train_set_SMILES = pd.DataFrame({'SMILES': train_set.SMILES})
test_set_SMILES = pd.DataFrame({'SMILES': test_set.SMILES})

features = pd.read_csv(config["BERT_features_file"])
features = features.drop_duplicates(subset = "SMILES").reset_index(drop = True)
train_X = pd.merge(train_set_SMILES, features, on = "SMILES", how = "inner")
train_X = train_X.drop(columns= 'SMILES').values

test_X = pd.merge(test_set_SMILES, features, on = "SMILES", how = "inner")
test_X = test_X.drop(columns= 'SMILES').values
print("After merging", train_X.shape, test_X.shape)


# In[6]:


if config['Final_model']:
    train_X, train_y, train_ids = train_X, train_set[config["selected_tasks"]].values, train_set_SMILES.values
    val_X, val_y, val_ids = test_X, test_set[config["selected_tasks"]].values, test_set_SMILES.values
    fold_list = [0]
else:
    # genrate stratified random folds
    fold_data = get_stratified_folds(train_X, 
                                train_set[config["selected_tasks"]].values,
                                train_set_SMILES.values, 
                                num_of_folds = 5, 
                                config = config)
    fold_list = [0,1,2,3,4]


# In[7]:

config["loss_type"] = "Focal_loss" #"Focal_loss",# "BCE","Focal_loss_v2"
l2_lambda_list = [1e-3]#[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
dropout_p_list = [0.2]
alpha_list = [0.0]
gamma_list = [0.0]


# In[8]:


for alpha in alpha_list:
    config["alpha"] = alpha

    for gamma in gamma_list:
        config["gamma"] = gamma

        for l2_lambda in l2_lambda_list:
            config["optm_l2_lambda"] = l2_lambda

            for dropout_p in dropout_p_list:
                config["dropout_p"] = dropout_p

                y_true = pd.DataFrame()
                y_pred = pd.DataFrame()

                for fold in fold_list:
                    config["fold"] = fold  
                    config["model_name"] = rf's{config["seed"]}_alpha_{config["alpha"]}_gamma_{config["gamma"]}_loss_type_{config["loss_type"]}_λ{config["optm_l2_lambda"]}_f{config["fold"]}'
        
                    if config['Final_model']:
                        train_X, train_y, train_ids = train_X,train_set[config["selected_tasks"]].values,train_set_SMILES.values
                        val_X, val_y, val_ids = test_X,test_set[config["selected_tasks"]].values,test_set_SMILES.values

                    else:
                        # get fold train, val data
                        train_X, train_y, train_ids = fold_data[fold]['train']['X'], fold_data[fold]['train']['y'], fold_data[fold]['train']['ids']
                        val_X, val_y, val_ids = fold_data[fold]['val']['X'], fold_data[fold]['val']['y'], fold_data[fold]['val']['ids']
                    
                    train_dataloader = DataLoader(dataloader_for_numpy(train_X, train_y, x_type = 'Fingerprints'), 
                                                                batch_size=config["batch_size"],
                                                                pin_memory=False,
                                                                num_workers=1, 
                                                                shuffle = True,
                                                                persistent_workers=True)

                    val_dataloader = DataLoader(dataloader_for_numpy(val_X, val_y, x_type = 'Fingerprints'),
                                                                batch_size=config["batch_size"], 
                                                                pin_memory=False,
                                                                shuffle = False,
                                                                num_workers=1,
                                                                persistent_workers=True) 
                      

                    config["training_steps"] = len(train_dataloader)
                    trained_model, run, trainer = wandb_init_model(model = Vanilla_MLP_classifier, 
                                                                train_dataloader = train_dataloader,
                                                                val_dataloader =val_dataloader,
                                                                config = config, 
                                                                model_type = 'MLP')
                
                    model = trained_model.eval()
                    if config["save_predicitons"]:
                        # get model predictions
                        y_df, y_hat_df = get_model_predictions_MT(model, val_dataloader, config, val_ids)
                        y_true = pd.concat([y_true, y_df], axis = 0)
                        y_pred = pd.concat([y_pred, y_hat_df], axis = 0)

                # after 5 folds
                if config["save_predicitons"]:
                    data_dir = config["metadata_dir"] + "predicitons/"
                    result_dir = config["metadata_dir"] + "Results/"  
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                        os.makedirs(result_dir)
                    # Sort w.r.t toxic observations
                    y_true_val, y_pred_val = y_true.reset_index(drop = True), y_pred.reset_index(drop = True)

                    # also save train pred
                    y_true_train, y_pred_train = get_model_predictions_MT(model, train_dataloader, config, train_ids)

                    # save predictions
                    name = config['model_name'].split('_f')[0] + '.csv'
                    y_true_val.to_csv(data_dir + 'y_true_val_' + name, index=False)
                    y_pred_val.to_csv(data_dir + 'y_pred_val_' + name, index=False)

                    y_true_train.to_csv(data_dir + 'y_true_train_' + name, index=False)
                    y_pred_train.to_csv(data_dir + 'y_pred_train_' + name, index=False)

                metrics = compute_binary_classification_metrics_MT(y_true = y_true_val[config['selected_tasks']].values, 
                                                                y_pred_proba = y_pred_val[config['selected_tasks']].values,
                                                                missing = 'nan')
                metrics.insert(0, 'Tasks', target_names)  
                metrics.to_csv(result_dir + f'val_metric_' + name, index=False)

                # delete all, also clear gpu memory
                del train_dataloader, val_dataloader, trained_model, run, trainer
                torch.cuda.empty_cache()
                gc.collect()

                print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
                gpu_memory_status = torch.cuda.memory_allocated() / (1024 ** 3)
                print("GPU Memory Status (after clearing):", gpu_memory_status)
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++')


# In[ ]:
print("script complete")




# In[ ]:





# In[ ]:




