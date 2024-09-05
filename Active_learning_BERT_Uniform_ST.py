#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np

import torch
import deepchem as dc
from pytorch_lightning import seed_everything

import wandb
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")

import torch.multiprocessing as mp
import gc
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.data_utils import scafoldsplit_train_test, convert_to_dataframe, get_initial_set_with_main_and_aux_samples, drop_unwanted_tasks
from utils.data_utils import convert_dataframe_to_dataloader

from utils.utils import wandb_init_model, compute_binary_classification_metrics_MT, active_learning_loop
from utils.model_utils import get_pred_with_uncertainities
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# In[2]:

#################################
#task_list = [0,1,2]
#task_list = [3,4,5]
#task_list = [6,7,8]
#task_list = [9,10,11]

task_list = [10] #1,7,8,10 
gpu = 0


for task in task_list:
#################################
    seed_list = [42,88, 100, 142]
    for seed in seed_list:


        config = {
                "seed": seed,
                # directories
                "project_name": "Uniform_BERT_pretrained_Tox21_trainset_v2",
                 ### Vanilla BALD ###
                #"project_name": "Tox21_BERT_Final_Uniform",
                #"metadata_dir": '/projects/home/mmasood1/trained_model_predictions/Tox21/Frozen_BERT/Main_Aux_tasks_setting/1_main_0_aux/',
                #"target_file": "/projects/home/mmasood1/arslan_data_repository/Tox21/complete_Tox21.csv",
                #"BERT_features_file":"/projects/home/mmasood1/arslan_data_repository/Tox21/Tox21_BERT_features.csv",
                
                ### Uniform_pretrained_Tox21_trainset ###
                #"project_name": "Uniform_BERT_pretrained_Tox21_trainset",
                #"metadata_dir": '/projects/home/mmasood1/trained_model_predictions/Tox21/BERT_pretrained_on_Tox21_trainset/Main_Aux_tasks_setting/1_main_0_aux/',
                #"target_file": "/projects/home/mmasood1/arslan_data_repository/Tox21/BERT_representations/BERT_pretrained_on_Tox21_trainset/Tox21_filtered_data.csv",
                #"BERT_features_file":"/projects/home/mmasood1/arslan_data_repository/Tox21/BERT_representations/BERT_pretrained_on_Tox21_trainset/Tox21_BERT_features_pretrained_on_Tox21_trainset_epoch_99.csv",

                ### BALD_BERT_pretrained_Tox21_trainset (v2)###
                "metadata_dir": '/projects/home/mmasood1/trained_model_predictions/Tox21/BERT_pretrained_on_Tox21_trainset_v2/Main_Aux_tasks_setting/1_main_0_aux/',
                "target_file": "/projects/home/mmasood1/arslan_data_repository/Tox21/BERT_representations/BERT_pretrained_on_Tox21_trainset/Tox21_filtered_data.csv",
                "BERT_features_file":"/projects/home/mmasood1/arslan_data_repository/Tox21/BERT_representations/BERT_pretrained_on_Tox21_trainset/Tox21_BERT_features_pretrained_on_Tox21_trainset_epoch_675.csv",
                
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
                "dropout_p": 0.3,
                "BatchNorm1d": True,
                "use_skip_connection": True,
            
                # training
                "optim": 'Adam',#SGD
                "lr_schedulers": "CosineAnnealingLR",
                "lr": 1e-3,
                "l2_lambda": 0.0,
                "optm_l2_lambda": 1e-2,
                "epochs": 110,
                "compute_metric_after_n_epochs": 5,
                "batch_size": 16,
                "EarlyStopping": False, 
                "pretrained_model": False,
                
                # loss
                "missing" : 'nan',
                "alpha": 0.0,
                "beta": 0.0,
                "gamma":0.0,

                "gpu": [gpu],
                "accelerator": "gpu",
                "return_trainer": True, 
                "save_predicitons" : True,
                "Final_model": False,

                # active learning
                "num_forward_passes": 20,
                "num_itterations": 200,
                "sampling_strategy": "uniform",
                "n_query":1,
                "main_task_samples": 100
            }
        if config["gpu"] == [0]:
            config["device"]  = torch.device("cuda:0")
        if config["gpu"] == [1]:
            config["device"]  = torch.device("cuda:1")
        if config["gpu"] == [2]:
            config["device"]  = torch.device("cuda:2")
        if config["gpu"] == [3]:
            config["device"]  = torch.device("cuda:3")
        # get targets information
        # get targets information

        data = pd.read_csv(config["target_file"])
        all_tasks = data.loc[:, "NR-AR":"SR-p53"].columns.tolist()
        '''
        config["main_task"] = ["NR-AR"]
        config["aux_task"]  = all_tasks.copy()
        config["aux_task"].remove(config["main_task"][0])
        config["main_task_index"] = all_tasks.index(config["main_task"][0])
        config["aux_task_index"] = [all_tasks.index(i) for i in config["aux_task"]]
        '''
        config["all_tasks"] = all_tasks
        config["main_task"] = [all_tasks[task]]
        config["aux_task"] = None #["SR-MMP"]
        config["main_task_index"] = 0
        config["aux_task_index"] = None #np.arange(1, len(config["aux_task"])+1)

        target_names = config["main_task"]# + config["aux_task"]
        config["project_name"] = config["project_name"] +"_"+ config["main_task"][0] #+"_"+ config["aux_task"][0]

        #config["main_task_samples"] = 498
        #config["aux_task_samples"] = 50

        config["num_of_tasks"] = len(target_names)
        config["selected_tasks"] = target_names

        config["sample_only_from_aux"] = False
        config["use_all_tasks_to_split"] = True

        config["loss_type"] = "BCE" #"Focal_loss",# "BCE","Focal_loss_v2"



        # In[3]:


        # Splitting by using deepchem
        np.random.seed(config["seed"])
        train_set, test_set = scafoldsplit_train_test(config, all_tasks = config["use_all_tasks_to_split"])
        if config["use_all_tasks_to_split"]:
            train_set = drop_unwanted_tasks(train_set, config)
            test_set = drop_unwanted_tasks(test_set, config)

        #config["main_task_samples"] = (train_set.y[:, config["main_task_index"]] == 1).sum() * 2
        initial_set, train_set = get_initial_set_with_main_and_aux_samples(train_set, config)

        randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
        pool_set, val_set = randomstratifiedsplitter.train_test_split(train_set,
                                                                    frac_train = 0.85,
                                                                    seed = config["seed"])

        print("train_set", sorted(np.nansum(train_set.y, axis=0)))
        print("test_set", sorted(np.nansum(test_set.y, axis=0)))
        print("pool_set", sorted(np.nansum(pool_set.y, axis=0)))
        print("val_set", sorted(np.nansum(val_set.y, axis=0)))
        print("initial_set", sorted(np.nansum(initial_set.y, axis=0)))


        # In[4]:


        # Who cares about deepchem data_object, trash it
        initial_set = convert_to_dataframe(initial_set, config["selected_tasks"])
        val_set = convert_to_dataframe(val_set, config["selected_tasks"])
        pool_set = convert_to_dataframe(pool_set, config["selected_tasks"])
        test_set = convert_to_dataframe(test_set, config["selected_tasks"])

        # mask labels of main_task in pool_set
        if config["sample_only_from_aux"]:
            pool_set.loc[:,config["main_task"]] = np.nan


        # In[5]:

        # make dir
        t_names = config["metadata_dir"] + config["sampling_strategy"] + "/" + config["main_task"][0]
        query_set_dir = t_names +"/query_set/"
        result_dir = t_names +"/Results/"
        config["model_weights_dir"] = t_names +"/model_weights/"
        os.makedirs(query_set_dir, exist_ok = True)
        os.makedirs(result_dir, exist_ok = True)
        os.makedirs(config["model_weights_dir"], exist_ok = True)

        file_path = query_set_dir + "initial_set.csv"
        with open(file_path, 'w') as file:
            initial_set.to_csv(file, index = False)

        for itteration in range(config["num_itterations"]):
            #if pool_set.iloc[:, 1:].count().sum() >= config["n_query"]:

            from utils.models import Vanilla_MLP_classifier
            seed_everything(seed = config["seed"])
            config["itteration"] = itteration
            config["model_name"] = rf'itteration_{config["itteration"]}_s{config["seed"]}_alpha_{config["alpha"]}_gamma_{config["gamma"]}_loss_type_{config["loss_type"]}_λ{config["optm_l2_lambda"]}'
            
        
            # if model already has been trained
            try:
                file_path = os.path.join(query_set_dir, f'query_set_{config["model_name"]}.csv')
                with open(file_path, 'r') as file:
                    query_set = pd.read_csv(file)
                _, updated_training_set, updated_poolset = active_learning_loop(trained_model = None,
                                                                                    pool_dataloader = None, 
                                                                                    initial_set = initial_set,
                                                                                    pool_set = pool_set, 
                                                                                    config = config,
                                                                                    query_set = query_set)
                del initial_set,pool_set
                initial_set = updated_training_set.copy()
                pool_set = updated_poolset.copy()
                print("Trained model already exists, no sample taken")
            except:

                train_set_main_active,train_set_main_inactive =  (initial_set[config["main_task"]] == 1).sum().values, (initial_set[config["main_task"]] == 0).sum().values
                if config["aux_task"] != None:
                    train_set_aux_active,train_set_aux_inactive = (initial_set[config["aux_task"]] == 1).sum().sum(), (initial_set[config["aux_task"]] == 0).sum().sum()
                else:
                    train_set_aux_active,train_set_aux_inactive = np.array([0]),np.array([0])    
                train_set_main_total = train_set_main_active + train_set_main_inactive
                train_set_aux_total = train_set_aux_active + train_set_aux_inactive
                
                # get dataloaders
                train_dataloader = convert_dataframe_to_dataloader(dataframe= initial_set, config = config, shuffle= True, drop_last = True)
                val_dataloader = convert_dataframe_to_dataloader(dataframe= val_set, config = config, shuffle= False, drop_last = False)
                test_dataloader = convert_dataframe_to_dataloader(dataframe= test_set, config = config, shuffle= False, drop_last = False)
                pool_dataloader = convert_dataframe_to_dataloader(dataframe= pool_set, config = config, shuffle= False, drop_last = False)

                # Train model
                config["training_steps"] = len(train_dataloader)
                trained_model, run, trainer = wandb_init_model(model = Vanilla_MLP_classifier, 
                                                                        train_dataloader = train_dataloader,
                                                                        val_dataloader =val_dataloader,
                                                                        config = config, 
                                                                        model_type = 'MLP')
                
                train_set_main_total = train_set_main_active + train_set_main_inactive
                train_set_aux_total = train_set_aux_active + train_set_aux_inactive
                wandb.log({"train_set_main_total":train_set_main_total.item(),
                        "train_set_main_active":train_set_main_active.item(),
                        "train_set_main_inactive":train_set_main_inactive.item()})
                
                wandb.log({"train_set_aux_total":train_set_aux_total.item(),
                        "train_set_aux_active":train_set_aux_active.item(),
                        "train_set_aux_inactive":train_set_aux_inactive.item()})
                
                wandb.finish()

                ###### Model Evaluation #############

                # Evaluation
                trained_model = trained_model.eval()
                targets, pred_mean, pred_var, all_pred = get_pred_with_uncertainities(test_dataloader, trained_model,
                                                                    n_samples=test_set.shape[0],
                                                                    n_classes=config["num_of_tasks"],
                                                                    cal_uncert=False,
                                                                    num_forward_passes= 1)
                metrics = compute_binary_classification_metrics_MT(targets, pred_mean, missing='nan')

                metrics = metrics.append(metrics.mean(), ignore_index=True)
                metrics.insert(0, 'Tasks', config["selected_tasks"] + ['mean'])

                # save metrics
                file_path = os.path.join(result_dir + f'/itteration_{itteration}_metrics_{config["model_name"]}.csv')
                with open(file_path, 'w') as file:
                    metrics.to_csv(file, index=False)
                print(metrics.mean())

                query_set, updated_training_set, updated_poolset = active_learning_loop(trained_model,
                                                                                    pool_dataloader, 
                                                                                    initial_set,
                                                                                    pool_set, 
                                                                                    config)
                # save query_set
                file_path = os.path.join(query_set_dir, f'query_set_{config["model_name"]}.csv')
                with open(file_path, 'w') as file:
                    query_set.to_csv(file, index=False)

                del initial_set,pool_set
                del train_dataloader,val_dataloader,test_dataloader,pool_dataloader, trained_model

                initial_set = updated_training_set.copy()
                pool_set = updated_poolset.copy()

                if config["gpu"] == [0]:
                    torch.cuda.empty_cache()
                if config["gpu"] == [1]:
                    with torch.cuda.device('cuda:1'):
                        torch.cuda.empty_cache()
                gc.collect()

                print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
                gpu_memory_status = torch.cuda.memory_allocated() / (1024 ** 3)
                print("GPU Memory Status (after clearing):", gpu_memory_status)
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++')

                active_procs = mp.active_children()
                # Loop through the list of active child processes and terminate them
                for proc in active_procs:
                    proc.terminate()
        #else:
            #print("All label exhausted, Terminating!!!")
            #break

print("####### Script completed ###############")

# In[ ]:





# In[ ]:





# In[ ]:




