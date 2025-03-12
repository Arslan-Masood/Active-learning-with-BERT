import torch.optim as optim
from typing import List, Union
from scipy.special import expit
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score

import numpy as np
import torch
import pandas as pd
import re

import logging
import math
from torch import Tensor

###########################################################
#  LRScheduler
###########################################################
class NoamLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: List[Union[float, int]], total_epochs: List[int], steps_per_epoch: int, init_lr: List[float], max_lr: List[float], final_lr: List[float]):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.current_step = 0

        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        print(self.warmup_steps, self.total_steps)
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

    def step(self, epoch=None):
        if epoch is not None:
            self.current_step = epoch * self.steps_per_epoch
        else:
            self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr = self.init_lr + self.current_step * self.linear_increment
        elif self.current_step <= self.total_steps:
            lr = self.max_lr * (self.exponential_gamma ** (self.current_step - self.warmup_steps))
        else:
            lr = self.final_lr

        self.optimizer.param_groups[0]['lr'] = lr


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_pred_with_uncertainities(dataloader, model, n_samples, n_classes, cal_uncert=False, num_forward_passes=1, device = None):

    dropout_predictions = []
    for i in range(num_forward_passes):
        torch.manual_seed(i)
        model = model.eval()
        model = model.to(device)
        if cal_uncert == True:
            enable_dropout(model)
        preds, targets = [], []
        for batch in dataloader:
            batch_x, batch_targets = batch
            with torch.no_grad():
                batch_preds = model(batch_x.to(device))
                batch_preds = batch_preds.cpu().detach().tolist()
                preds.extend(batch_preds)
                targets.extend(batch_targets.cpu().detach().tolist())
                 
        preds = expit(np.array(preds))
        targets = np.array(targets)
        dropout_predictions.append(preds.reshape(-1, n_samples, n_classes))

    dropout_predictions = np.concatenate(dropout_predictions, axis=0)
        
    pred_mean = np.mean(dropout_predictions, axis=0)
    pred_var = np.var(dropout_predictions, axis=0)

    return targets, pred_mean, pred_var, dropout_predictions


def BALD_acquisition_function(pred):
    '''
    pred: probability (repeats, mol, tasks)
    '''
    pred_mean = pred.mean(axis=0)
    epsilon = 1e-10
    # Calculating entropy across multiple MCD forward passes
    # input (mol, tasks) --> (mol, tasks)
    H = - pred_mean * np.log(pred_mean + epsilon)  # shape (n_samples,task)
    E_H = np.mean(-pred * np.log(pred + epsilon), axis=0)
    # Calculating mutual information across multiple MCD forward passes
    MI = H - E_H
    return MI

######################################################################3
# EPIG_MT
#######################################################################
def check(
    scores: Tensor, max_value: float = math.inf, epsilon: float = 1e-6, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        
        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")
    
    return scores

def conditional_epig_from_probs(probs_pool, probs_targ):
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [k, N_p, Cl_p]
        probs_targ: Tensor[float], [k, N_t, Cl_t]

    Returns:
        Tensor[float], [N_p, N_t, cl_p]
    """

    # Estimate the joint predictive distribution.
    probs_pool = probs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl_p, 1]
    probs_targ = probs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl_t]
    probs_pool_targ_joint = probs_pool * probs_targ  # [K, N_p, N_t, Cl_p, Cl_t]
    probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl_p, Cl_t]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=0)  # [N_p, 1, Cl_p, 1]
    probs_targ = torch.mean(probs_targ, dim=0)  # [1, N_t, 1, Cl_t]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl_p, Cl_t]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    nonzero_joint = probs_pool_targ_joint > 0  # [N_p, N_t, Cl_p, Cl_t]
    log_term = torch.clone(probs_pool_targ_joint)  # [N_p, N_t, Cl_p, Cl_t]
    log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])  # [N_p, N_t, Cl_p, Cl_t]
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl_p, Cl_t]

    #scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=-1)  # [N_p, N_t, Cl_p]
    return scores  # [N_p, N_t, cl_p]

def epig_from_conditional_scores(scores):
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t, cl_p]

    Returns:
        Tensor[float], [N_p,cl_p]
    """
    scores = torch.mean(scores, dim=1)  # [N_p,cl_p]
    scores = check(scores, score_type="EPIG")
    return scores  # [N_p,cl_p]

def EPIG_MT_acquisition_function(probs_pool, probs_targ):
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [K, N_p, Cl_p]
        probs_targ: Tensor[float], [K, N_t, Cl_t]

    Returns:
        Tensor[float], [N_p,cl_t]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t, cl_p]
    return epig_from_conditional_scores(scores)  # [N_p,cl_p]
#############################################################################################

def get_random_indices(pool_set, args):
    query_mol = np.random.choice(range(pool_set.shape[0]), size= args.n_query, replace=False)
    query_task = np.random.choice(range(len(args.selected_tasks)), size= args.n_query, replace=True)
    return (query_mol, query_task)

def get_top_indices(array_2d, topk):

    # Flatten the array and get indices of N largest values
    flat_indices = np.argpartition(array_2d.flatten(), -topk)[-topk:]

    # Sort the top N values and their corresponding indices
    sorted_indices = flat_indices[np.argsort(
        array_2d.flatten()[flat_indices])][::-1]

    # Convert flat indices to 2D indices
    indices_2d = np.unravel_index(sorted_indices, array_2d.shape)

    return indices_2d

def get_top_indices_from_aux_task(array_2d, topk, aux_task_number):
    '''
    # Single Task query from the selected task
    '''
    sorted_indices = np.argsort(-array_2d[:,aux_task_number])[:topk]
    task_array = np.ones_like(sorted_indices) * aux_task_number
    return (sorted_indices,task_array)


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

    num_tasks = y_true.shape[1]  # Get the number of tasks
    metrics_list = []

    for i in range(num_tasks):
        y_true_task = y_true[:, i]
        y_pred_proba_task = y_pred_proba[:, i]

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
        

        metrics_list.append(metrics_task)
    metrics_df = pd.DataFrame(metrics_list)
    col = ['balanced_acc', 'f1_score','specificity','sensitivity', 'roc_auc','AUPR', 'average_precision']
    
    return metrics_df[col]


def pretrained_model(model, args):
    debug = info = print

    # Load model and args

    state = torch.load(args.pretrained_dir, map_location=lambda storage, loc: storage)
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