
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
import os
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")


# In[2]:


from utils.data_utils import scafoldsplit_train_test, dataloader_for_numpy
from utils.utils import wandb_init_model
from utils.models import Custom_Chemprop

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.model_utils import pretrained_model
from utils.model_utils import get_chemprop_pred, compute_binary_classification_metrics_MT


# In[3]:


TrainArgs.project_name = 'DMPNN_Tox21'
TrainArgs.model_name = 'Trial'

TrainArgs.target_file = "/projects/home/mmasood1/arslan_data_repository/Tox21/complete_Tox21.csv"
TrainArgs.input_dim = 1024
TrainArgs.train_frac = 0.8

TrainArgs.pretrained_dir = "/projects/home/mmasood1/Model_weights/invitro/Chemprop/fold_0/fold_0/model_0/"
TrainArgs.model_weights_dir = "/projects/home/mmasood1/Model_weights/preclinical_clinical/chemprop/"
TrainArgs.metadata_dir = '/projects/home/mmasood1/trained_model_predictions/Tox21/Chemprop/'
TrainArgs.pretrained_model = False

TrainArgs.depth = 3
TrainArgs.hidden_size = 300
TrainArgs.ffn_num_layers = 2
TrainArgs.ffn_hidden_size = 300
TrainArgs.num_of_tasks = None
TrainArgs.use_input_features = False
TrainArgs.dropout = 0.2
TrainArgs.batch_size = 50
TrainArgs.adding_bond_types = True
TrainArgs.atom_descriptors_size = 0

TrainArgs.scheduler_type = 'ReduceLROnPlateau'
TrainArgs.warmup_epochs = 2
TrainArgs.epochs = 100
TrainArgs.init_lr = 1e-4
TrainArgs.max_lr = 1e-3
TrainArgs.final_lr = 1e-4
TrainArgs.weight_decay = 1e-5
TrainArgs.loss_function = "binary_cross_entropy"
TrainArgs.seed = 42

TrainArgs.accelerator = 'gpu'
TrainArgs.EarlyStopping = True
TrainArgs.return_trainer = True
TrainArgs.device = torch.device("cuda")
TrainArgs.compute_metrics_during_training = False
TrainArgs.num_forward_passes = 10
TrainArgs.calc_uncert = True
TrainArgs.compute_metric_after_n_epochs = 5



# In[4]:


args = TrainArgs
args.dataset_type = 'classification'
args.metric = 'auc'
args.is_atom_bond_targets = False
args.use_target_weights = False
args.missing_label_representation = 'nan'


# In[5]:


# get targets information
data = pd.read_csv(args.target_file)
target_names = data.loc[:,"NR-AR":"SR-p53"].columns.tolist()

args.num_of_tasks = len(target_names)
args.selected_tasks = target_names


# In[6]:

np.random.seed(0)
seed_everything(seed = args.seed)

train_set, test_set = scafoldsplit_train_test(target_file = args.target_file,
                              selected_tasks = args.selected_tasks, 
                              FP_size = args.input_dim,
                              train_frac = args.train_frac,)

print(sorted(np.nansum(train_set.y, axis = 0)))
print(sorted(np.nansum(test_set.y, axis = 0)))


# In[7]:


# use complete set
train_X, train_y = train_set.ids, train_set.y
val_X, val_y = test_set.ids, test_set.y

train_dataloader = DataLoader(dataloader_for_numpy(train_X, train_y, x_type = 'SMILES'), 
                                            batch_size=args.batch_size,
                                            pin_memory=False,
                                            num_workers=4, 
                                            shuffle = True,
                                            persistent_workers=True)

val_dataloader = DataLoader(dataloader_for_numpy(val_X, val_y, x_type = 'SMILES'),
                                                    batch_size=args.batch_size, 
                                                    pin_memory=False,
                                                    shuffle = False,
                                                    num_workers=4,
                                                    persistent_workers=True)
args.steps_per_epoch = len(train_dataloader)
args.model_name = f'd{args.depth}_MPN_h{args.hidden_size}_ffn_h{args.ffn_hidden_size}_DO{args.dropout}'

trained_model, run, trainer = wandb_init_model(Custom_Chemprop, 
                                      args, 
                                      train_dataloader,
                                      val_dataloader,
                                      model_type = 'chemprop')

checkpoint_callback = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)][0]
metric_to_optimize = checkpoint_callback.best_model_score.item()

wandb.finish()

#models_chkpoints = [checkpoint_callback.best_model_path, checkpoint_callback.last_model_path]
#model_type = ["best_model", "last_model"]

data_dir = args.metadata_dir + "predicitons/"
result_dir = args.metadata_dir + "Results/"

#for i, chk_path in enumerate(models_chkpoints):
#    print(i)
#    args.pretrained_dir = chk_path
#    trained_model = pretrained_model(Custom_Chemprop,args)
model = trained_model.eval()

targets, preds, pred_var = get_chemprop_pred(val_dataloader, model, 
                                n_samples= val_y.shape[0], 
                                n_classes= val_y.shape[1],
                                cal_uncert= args.calc_uncert, 
                                num_forward_passes = args.num_forward_passes)

metrics = compute_binary_classification_metrics_MT(targets, preds, missing = 'nan')
metrics = metrics.append(metrics.mean(), ignore_index= True)
metrics.insert(0, 'Tasks', args.selected_tasks + ['mean'])

obs = pd.DataFrame(targets)
obs.columns = args.selected_tasks
obs.insert(0, 'SMILES',test_set.ids)

pred = pd.DataFrame(preds)
pred.columns = args.selected_tasks
pred.insert(0, 'SMILES',test_set.ids)

pred_var = pd.DataFrame(pred_var)
pred_var.columns = args.selected_tasks
pred_var.insert(0, 'SMILES',test_set.ids)

label = 'test'

obs.to_csv(data_dir + f'{label}_obs.csv', index=False)
pred.to_csv(data_dir + f'{label}_pred.csv', index=False)
pred_var.to_csv(data_dir + f'{label}_pred_var.csv', index=False)
metrics.to_csv(result_dir + f'{label}_metrics.csv', index=False)
    
print("script completed")