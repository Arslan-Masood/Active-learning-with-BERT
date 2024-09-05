#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from chemprop.args import TrainArgs
from pytorch_lightning import seed_everything


import wandb
import deepchem as dc
import os
os.environ["WANDB_SILENT"] = "true"
wandb.login(key="27edf9c66b032c03f72d30e923276b93aa736429")

from utils.data_utils import scafoldsplit_train_test, dataloader_for_numpy,convert_to_dataframe, convert_dataframe_to_dataloader
from utils.data_utils import get_random_query_set, get_query_set, update_training_set, remove_queried_index_from_pool_set
from utils.utils import wandb_init_model
from utils.models import Custom_Chemprop

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.model_utils import pretrained_model, BALD_acquisition_function, get_top_indices, get_random_indices
from utils.model_utils import get_chemprop_pred, compute_binary_classification_metrics_MT


# In[2]:


TrainArgs.project_name = 'Tox21_BALD_v4'
TrainArgs.model_name = 'Trial'

TrainArgs.target_file = "/projects/home/mmasood1/arslan_data_repository/Tox21/complete_Tox21.csv"
TrainArgs.BERT_features_file = "/projects/home/mmasood1/arslan_data_repository/Tox21/Tox21_BERT_features.csv"

TrainArgs.input_dim = 1024
TrainArgs.train_frac = 0.8

TrainArgs.pretrained_dir = "/projects/home/mmasood1/Model_weights/invitro/Chemprop/fold_0/fold_0/model_0/"
TrainArgs.model_weights_dir = "/projects/home/mmasood1/Model_weights/preclinical_clinical/chemprop/"
TrainArgs.metadata_dir = '/projects/home/mmasood1/Active_learning_models_predictions/Tox21/Chemprop/v4/BALD/'
TrainArgs.pretrained_model = False

TrainArgs.depth = 3
TrainArgs.hidden_size = 300
TrainArgs.ffn_num_layers = 2
TrainArgs.ffn_hidden_size = 300
TrainArgs.num_of_tasks = None
TrainArgs.use_input_features = False
TrainArgs.dropout = 0.25
TrainArgs.batch_size = 50
TrainArgs.adding_bond_types = True
TrainArgs.atom_descriptors_size = 0

TrainArgs.scheduler_type = 'ReduceLROnPlateau'
TrainArgs.warmup_epochs = 2
TrainArgs.epochs = 5
TrainArgs.init_lr = 1e-4
TrainArgs.max_lr = 1e-3
TrainArgs.final_lr = 1e-4
TrainArgs.weight_decay = 0

TrainArgs.loss_function = "binary_cross_entropy"
TrainArgs.seed = 42

TrainArgs.accelerator = 'gpu'
TrainArgs.EarlyStopping = False
TrainArgs.return_trainer = True
TrainArgs.device = torch.device("cuda")
TrainArgs.compute_metrics_during_training = True

TrainArgs.num_forward_passes = 20
TrainArgs.n_query = 100
TrainArgs.num_itterations = 10000
TrainArgs.sampling_strategy = "BALD"
TrainArgs.seed = 42
TrainArgs.compute_metric_after_n_epochs = 5

args = TrainArgs
args.dataset_type = 'classification'
args.metric = 'auc'
args.is_atom_bond_targets = False
args.use_target_weights = False
args.missing_label_representation = 'nan'


# In[3]:


args = TrainArgs
args.dataset_type = 'classification'
args.metric = 'auc'
args.is_atom_bond_targets = False
args.use_target_weights = False
args.missing_label_representation = 'nan'

args.features_type = "FP"
args.FP_size = 1024


# In[4]:


# get targets information
data = pd.read_csv(args.target_file)
target_names = data.loc[:, "NR-AR":"SR-p53"].columns.tolist()

args.num_of_tasks = len(target_names)
args.selected_tasks = target_names


# In[5]:


np.random.seed(args.seed)
seed_everything(seed = args.seed)
train_set, test_set = scafoldsplit_train_test(args)

randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()

pool_set, val_set, initial_set = randomstratifiedsplitter.train_valid_test_split(train_set,
                                                                                 frac_train=0.85,
                                                                                 frac_valid=0.134,
                                                                                 frac_test=0.016,
                                                                                 seed=42)
'''
pool_set, val_set, initial_set = randomstratifiedsplitter.train_valid_test_split(train_set,
                                                                                 frac_train=0.7,
                                                                                 frac_valid=0.2,
                                                                                 frac_test=0.1,
                                                                                 seed=42)
'''
print("train_set", sorted(np.nansum(train_set.y, axis=0)))
print("test_set", sorted(np.nansum(test_set.y, axis=0)))
print("pool_set", sorted(np.nansum(pool_set.y, axis=0)))
print("val_set", sorted(np.nansum(val_set.y, axis=0)))
print("initial_set", sorted(np.nansum(initial_set.y, axis=0)))
print(pool_set.y.shape, initial_set.y.shape, val_set.y.shape)


# In[6]:


# Who cares about deepchem data_object, trash it
initial_set = convert_to_dataframe(initial_set, args.selected_tasks)
val_set = convert_to_dataframe(val_set, args.selected_tasks)
pool_set = convert_to_dataframe(pool_set, args.selected_tasks)
test_set = convert_to_dataframe(test_set, args.selected_tasks)


# In[7]:


'''
def Active_learning_loop(model,args,
                         initial_set,
                         pool_set,
                         val_set,
                         test_set):
'''
for itteration in range(args.num_itterations):

    from utils.models import Custom_Chemprop
    model = Custom_Chemprop

    ##### initiate dataloader ##########
    train_dataloader = convert_dataframe_to_dataloader(dataframe= initial_set, args = args, shuffle= True)
    val_dataloader = convert_dataframe_to_dataloader(dataframe= val_set, args = args, shuffle= False)
    test_dataloader = convert_dataframe_to_dataloader(dataframe= test_set, args = args, shuffle= False)
    pool_dataloader = convert_dataframe_to_dataloader(dataframe= pool_set, args = args, shuffle= False)

    ##### Model training #############
    args.steps_per_epoch = len(train_dataloader)
    args.model_name = f'itteration_{itteration}_d{args.depth}_MPN_h{args.hidden_size}_ffn_h{args.ffn_hidden_size}_DO{args.dropout}'

    trained_model, run, trainer = wandb_init_model(model,
                                                args,
                                                train_dataloader,
                                                val_dataloader,
                                                model_type='chemprop')
    checkpoint_callback = [
        cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)][0]
    metric_to_optimize = checkpoint_callback.best_model_score.item()
    wandb.finish()

    ###### Model Evaluation #############
    query_set_dir = args.metadata_dir + "query_set"
    result_dir = args.metadata_dir

    os.makedirs(query_set_dir, exist_ok = True)
    os.makedirs(result_dir, exist_ok = True)

    trained_model = trained_model.eval()
    targets, pred_mean, pred_var, all_pred = get_chemprop_pred(test_dataloader, trained_model,
                                                        n_samples=test_set.shape[0],
                                                        n_classes=len(args.selected_tasks),
                                                        cal_uncert=False,
                                                        num_forward_passes=1)

    # compute test metrics
    metrics = compute_binary_classification_metrics_MT(
        targets, pred_mean, missing='nan')
    print(metrics.mean())
    metrics = metrics.append(metrics.mean(), ignore_index=True)
    metrics.insert(0, 'Tasks', args.selected_tasks + ['mean'])
    metrics.to_csv(result_dir + f'itteration_{itteration}_metrics.csv', index=False)

    ####### Uncertainity estmation ################
    if args.sampling_strategy == "BALD":
        targets, pred_mean, pred_var, all_pred = get_chemprop_pred(pool_dataloader, trained_model,
                                                        n_samples=pool_set.shape[0],
                                                        n_classes=len(args.selected_tasks),
                                                        cal_uncert=True,
                                                        num_forward_passes=args.num_forward_passes)
        acquisition = BALD_acquisition_function(all_pred)
        
        # We should not query with missing labels, so hide it
        nan_mask = ~np.isnan(pool_set[args.selected_tasks].values)
        acquisition = acquisition * nan_mask
        print(np.max(acquisition))

        # Get location of TopGuns
        top_indices = get_top_indices(acquisition, args.n_query)
        query_set = get_query_set(pool_set, top_indices)

    if args.sampling_strategy == "uniform":
        query_set = get_random_query_set(pool_set, args)

    ##########  updated dataset ##################
    updated_training = update_training_set(initial_set, query_set)
    updated_poolset = remove_queried_index_from_pool_set(pool_set, query_set, args)

    initial_counts = initial_set.iloc[:, 1:].count().sum()
    query_counts = query_set.iloc[:, 1:].count().sum()
    updated_training_counts = updated_training.iloc[:, 1:].count().sum()

    initial_poolset_counts = pool_set.iloc[:, 1:].count().sum()
    updated_poolset_counts = updated_poolset.iloc[:, 1:].count().sum()

    print(
            "initial_counts", initial_counts, 
            "initial_poolset_count", initial_poolset_counts, 
            "query_counts", query_counts, 
            "updated_training_counts", updated_training_counts, 
            "updated_poolset_counts", updated_poolset_counts)
    
    # Use an assertion to check if the counts are equal
    assert updated_training_counts == initial_counts + query_counts, "Training_count,Queryset counts are not equal"
    assert updated_poolset_counts == initial_poolset_counts - query_counts, "Poolset, Queryset count are not equal"
    query_set.to_csv(query_set_dir + f'itteration_{itteration}_query_set.csv', index=False)

    del initial_set,pool_set
    del train_dataloader,val_dataloader,test_dataloader,pool_dataloader, trained_model, model,Custom_Chemprop
    
    initial_set = updated_training.copy()
    pool_set = updated_poolset.copy()


# In[ ]:







# In[ ]:





