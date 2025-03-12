
import re
import pandas as pd
import numpy as np
import wandb
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def wandb_init_model(model, config, train_dataloader,val_dataloader, model_type):
    if val_dataloader == None:
        limit_val_batches = 0.0
    else:
        limit_val_batches = 1.0
    # Init our model
    if model_type == 'chemprop':
        run = wandb.init(
                        project= config.project_name,
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = None,
                        name = config.model_name,
                        settings=wandb.Settings(start_method="fork"))
        
        default_root_dir = config.model_weights_dir
        use_pretrained_model = config.pretrained_model
        use_EarlyStopping = config.EarlyStopping
        max_epochs = config.epochs
        accelerator =config.accelerator
        return_trainer = config.return_trainer
        print(max_epochs)
    else:
        run = wandb.init(
                        project= config["project_name"],
                        dir = '/projects/home/mmasood1/Model_weights',
                        entity="arslan_masood", 
                        reinit = True, 
                        config = config,
                        name = config["model_name"],
                        settings=wandb.Settings(start_method="fork"))
        
        default_root_dir = config["model_weights_dir"]
        use_pretrained_model = config["pretrained_model"]
        use_EarlyStopping = config["EarlyStopping"]
        max_epochs = config["epochs"]
        accelerator =config["accelerator"]
        return_trainer = config["return_trainer"]

    if use_pretrained_model:
        model = pretrained_model(model,config)
    else:
        model = model(config)
    wandb_logger = WandbLogger()
    #wandb_logger.watch(model, log="all",log_freq=1)
    
    if use_EarlyStopping == True:
        callback = [EarlyStopping(
                                monitor='val_BCE_loss',
                                min_delta=1e-5,
                                patience=10,
                                verbose=False,
                                mode='min'
                                )]
    else:
        callback = []

    checkpoint_callback = ModelCheckpoint(
    monitor='val_BCE_non_weighted',  # Metric to monitor for saving the best model
    mode='min',          # Minimize the monitored metric
    dirpath= default_root_dir,  # Directory to store checkpoints
    filename='model-{epoch:02d}-{val_BCE_loss:.2f}',  # Checkpoint filename format
    #filename=config['chkp_file_name'],  # Checkpoint filename format
    save_top_k=1,
    save_last = True)
    callback.append(checkpoint_callback)


    trainer = Trainer(
        callbacks=callback,
        max_epochs= int(max_epochs),
        accelerator= accelerator, 
        #devices= config['gpu'],
        #limit_val_batches = 5,
        #limit_train_batches= 5,
        #precision=16,
        enable_progress_bar = True,
        #profiler="simple",
        enable_model_summary=True,
        logger=wandb_logger,
        default_root_dir=default_root_dir)

    # model fitting 
    trainer.fit(model, 
                train_dataloaders=train_dataloader,
                val_dataloaders =val_dataloader,
                )
    if return_trainer:
        return model, run, trainer
    else:
        return model, run
    

#####################################################################################3
# get pretrained model
################################################################################
def pretrained_model(model, args):
    debug = info = print

    # Load model and args
    state = torch.load(args.pretrained_dir + 'model.pt', map_location=lambda storage, loc: storage)
    loaded_state_dict = state["state_dict"]

    # Remove last layer
    loaded_state_dict = {key: value for key, value in list(loaded_state_dict.items())[:-2]}

    model = model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)
    return model

#####################################################################################3
# get model predictions
################################################################################
def get_model_predictions_MT(model, selected_dataloader, config, ids):
    
    y_true_list = []
    y_pred_list = []
    for batch in selected_dataloader:
        x, y = batch
        y_hat = model(x)
        y_true_list.append(y.cpu())
        y_pred_list.append(y_hat.cpu())
    
    y = torch.cat(y_true_list, dim=0)
    y_hat = torch.cat(y_pred_list, dim=0)

    if config["num_of_tasks"] > 1:
        y = pd.DataFrame(y.cpu().detach().numpy())
        y_hat = pd.DataFrame(y_hat.cpu().detach().numpy())
        y.columns = config['selected_tasks']
        y_hat.columns = config['selected_tasks']
    else:
        y = pd.DataFrame({config["selected_tasks"]: y.cpu().detach().numpy()})
        y_hat = pd.DataFrame({config["selected_tasks"]: y_hat.cpu().detach().numpy().reshape(-1)})

    y.insert(0,'SMILES', ids)
    y_hat.insert(0,'SMILES', ids)
    return y,y_hat

#####################################################################################3
# Compute compute_binary_classification_metrics: Multitask
######################################################################################
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, f1_score
def prob_to_labels(pred, threshold):
	    return (pred >= threshold).astype('int')

def compute_binary_classification_metrics_MT(y_true, y_pred_proba, 
                                             missing):
    """
    Compute various metrics for binary classification.
    
    Parameters:
        y_true (array-like): Binary labels (0 or 1).
        y_pred_proba (array-like): Predictive probabilities for the positive class.
        threshold (float, optional): Threshold value for classification. Default is 0.5.
    
   Returns:
        pandas.DataFrame: DataFrame containing the computed metrics for each task (accuracy, ROC AUC, average precision, MCC, F1-score, random precision, gain in average precision).
    """
    try:
        num_tasks = y_true.shape[1]  # Get the number of tasks
    except:
        num_tasks = 1
    metrics_list = []

    for i in range(num_tasks):
        if num_tasks > 1:
            y_true_task = y_true[:, i]
            y_pred_proba_task = y_pred_proba[:, i]
        else:
            y_true_task = y_true
            y_pred_proba_task = y_pred_proba
            
        # Apply masking
        if missing == 'nan':
            mask = ~np.isnan(y_true_task)
        if missing == -1:
            mask = (y_true_task != -1)

        y_true_task = y_true_task[mask]
        y_pred_proba_task = y_pred_proba_task[mask]


        metrics_task = {}
        try:
            # ROC AUC
            fpr, tpr, th = roc_curve(y_true_task, y_pred_proba_task)
            metrics_task['roc_auc'] = auc(fpr, tpr)

            # Balanced accuracy
            balanced_accuracy = (tpr + (1 - fpr)) / 2
            metrics_task['balanced_acc'] = np.max(balanced_accuracy)
            
            # sensitivity, specificity
            optimal_threshold_index = np.argmax(balanced_accuracy)
            optimal_threshold = th[optimal_threshold_index]
            metrics_task['sensitivity'] = tpr[optimal_threshold_index]
            metrics_task['specificity'] = 1 - fpr[optimal_threshold_index]

        except:
            metrics_task['roc_auc'] = np.nan
            metrics_task['sensitivity']= np.nan
            metrics_task['specificity']= np.nan
        try:
            precision, recall, thresholds = precision_recall_curve(y_true_task, y_pred_proba_task)
            metrics_task['AUPR'] = auc(recall, precision)
            f1 = [f1_score(y_true_task, prob_to_labels(y_pred_proba_task, t)) for t in thresholds]
            metrics_task['f1_score'] = np.max(f1)

            metrics_task['average_precision'] = average_precision_score(y_true_task, y_pred_proba_task)
        except:
            metrics_task['AUPR'] = np.nan
            metrics_task['f1_score'] = np.nan
        
        try:
            # calibration metrics
            metrics_task["ECE"] = compute_ece(y_true_task, y_pred_proba_task, n_bins=10, equal_intervals = True)
            metrics_task["ACE"] = compute_ece(y_true_task, y_pred_proba_task, n_bins=10, equal_intervals = False)
        except:
            metrics_task['ECE'] = np.nan
            metrics_task['ACE'] = np.nan

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 
           'roc_auc','AUPR', 'average_precision','ECE','ACE']
    
    return metrics_df[col]

################################################3
# Calibration metrics
################################################
def compute_ece(y_true, y_prob, n_bins=10, equal_intervals = True):
    # Calculate bin boundaries
    if equal_intervals == True: # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:                       # ACE
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    
    # Calculate bin indices
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    
    ece = 0
    total_samples = len(y_true)
    
    # Calculate ECE
    for bin_idx in range(n_bins):
        # Filter samples within the bin
        bin_mask = bin_indices == bin_idx
        bin_samples = np.sum(bin_mask)
        
        if bin_samples > 0:
            # Calculate accuracy and confidence for the bin
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_prob[bin_mask])
        
            # Update ECE
            ece += (bin_samples / total_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return ece

################################################3
# Calibration metrics
################################################
from utils.model_utils import BALD_acquisition_function, EPIG_MT_acquisition_function, get_top_indices
from utils.data_utils import get_random_query_set, get_query_set, update_training_set, remove_queried_index_from_pool_set
from utils.model_utils import get_pred_with_uncertainities

def active_learning_loop(trained_model,
                         pool_dataloader, 
                         initial_set,
                         pool_set, 
                         config,
                         query_set = None,
                         test_dataloader = None,
                         test_set = None):
    if query_set is None:
        ####### Uncertainity estmation ################
        if config["sampling_strategy"] == "BALD":
            _, _, _, all_pred = get_pred_with_uncertainities(pool_dataloader, trained_model,
                                                            n_samples=pool_set.shape[0],
                                                            n_classes=config["num_of_tasks"],
                                                            cal_uncert=True,
                                                            num_forward_passes=config["num_forward_passes"],
                                                            device = config["device"])
            acquisition = BALD_acquisition_function(all_pred)
            
            # We should not query with missing labels, so hide it
            nan_mask = ~np.isnan(pool_set[config["selected_tasks"]].values)
            acquisition = acquisition * nan_mask

            print(np.max(acquisition))

            # Get location of TopGuns
            top_indices = get_top_indices(acquisition, config["n_query"])
            query_set = get_query_set(pool_set, top_indices)

        if config["sampling_strategy"] == "EPIG_MT":
            print("######### EPIG SAMPLING ############")
            _, _, _, pool_pred = get_pred_with_uncertainities(pool_dataloader, trained_model,
                                                            n_samples=pool_set.shape[0],
                                                            n_classes=config["num_of_tasks"],
                                                            cal_uncert=True,
                                                            num_forward_passes=config["num_forward_passes"],
                                                            device = config["device"])

            pool_pred = torch.from_numpy(pool_pred)

            _,_, _, test_pred = get_pred_with_uncertainities(test_dataloader, trained_model,
                                                            n_samples=test_set.shape[0],
                                                            n_classes=config["num_of_tasks"],
                                                            cal_uncert=True,
                                                            num_forward_passes=config["num_forward_passes"],
                                                            device = config["device"])
            test_pred = torch.from_numpy(test_pred)

            # we are intereseted in one task
            test_pred_main_task = test_pred[:,:,config["main_task_index"]]
            test_pred_main_task = torch.unsqueeze(test_pred_main_task, dim = 2)
            
            acquisition = EPIG_MT_acquisition_function(pool_pred, test_pred_main_task)
            acquisition = acquisition.detach().numpy()
            
            # We should not query with missing labels, so hide it
            nan_mask = ~np.isnan(pool_set[config["selected_tasks"]].values)
            acquisition = acquisition * nan_mask
            print(np.max(acquisition))

            # Get location of TopGuns
            top_indices = get_top_indices(acquisition, config["n_query"])
            query_set = get_query_set(pool_set, top_indices)

        if config["sampling_strategy"] == "uniform":
            query_set = get_random_query_set(pool_set, config)  
    else:
        print("No sample taken")      
    ##########  updated dataset ##################
    updated_training_set = update_training_set(initial_set, query_set)
    updated_poolset = remove_queried_index_from_pool_set(pool_set, query_set, config)

    initial_counts = initial_set.iloc[:, 1:].count().sum()
    query_counts = query_set.iloc[:, 1:].count().sum()
    updated_training_counts = updated_training_set.iloc[:, 1:].count().sum()

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
    return query_set, updated_training_set, updated_poolset