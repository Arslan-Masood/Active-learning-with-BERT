import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
import deepchem as dc
from pytorch_lightning import seed_everything
from utils.data_utils import scafoldsplit_train_test, convert_to_dataframe, drop_unwanted_tasks, get_initial_set_with_main_and_aux_samples
from utils.utils import active_learning_loop

def get_config_from_args():
    parser = argparse.ArgumentParser(description='Active Learning Config')
    
    # Add arguments corresponding to all config parameters
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--project_name', type=str, required=True, help='Project name')
    parser.add_argument('--metadata_dir', type=str, required=True, help='Directory for metadata')
    parser.add_argument('--target_file', type=str, required=True, help='Path to target file')
    parser.add_argument('--bert_features_file', type=str, required=True, help='Path to BERT features file')
    parser.add_argument('--pos_weights', type=str, required=True, help='Path to positive class weights file')
    parser.add_argument('--class_weights', type=str, required=True, help='Path to class weights file')
    parser.add_argument('--features_type', type=str, required=True, choices=['BERT', 'other'], help='Type of features')
    parser.add_argument('--fp_size', type=int, required=True, help='Fingerprint size')
    parser.add_argument('--train_frac', type=float, required=True, help='Fraction of data to use for training')
    parser.add_argument('--input_dim', type=int, required=True, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension')
    parser.add_argument('--depth', type=int, required=True, help='Number of layers in the model')
    parser.add_argument('--dropout_p', type=float, required=True, help='Dropout probability')
    parser.add_argument('--batchnorm1d', type=bool, required=True, help='Whether to use BatchNorm1d')
    parser.add_argument('--use_skip_connection', type=bool, required=True, help='Whether to use skip connection')
    parser.add_argument('--optim', type=str, required=True, choices=['Adam', 'SGD'], help='Optimizer type')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--lr_schedulers', type=str, required=True, help='Learning rate scheduler')
    parser.add_argument('--l2_lambda', type=float, required=True, help='L2 regularization lambda')
    parser.add_argument('--optm_l2_lambda', type=float, required=True, help='Optimizer L2 regularization lambda')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--compute_metric_after_n_epochs', type=int, required=True, help='Epoch interval to compute metrics')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--early_stopping', type=bool, required=True, help='Use early stopping')
    parser.add_argument('--pretrained_model', type=bool, required=True, help='Use pretrained model')
    parser.add_argument('--missing', type=str, required=True, help='Missing value handling method')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha value for loss function')
    parser.add_argument('--beta', type=float, required=True, help='Beta value for loss function')
    parser.add_argument('--gamma', type=float, required=True, help='Gamma value for loss function')
    parser.add_argument('--gpu', type=int, required=True, help='GPU index to use')
    parser.add_argument('--accelerator', type=str, required=True, help='Accelerator type (e.g., gpu)')
    parser.add_argument('--num_forward_passes', type=int, required=True, help='Number of forward passes for uncertainty estimation')
    parser.add_argument('--num_iterations', type=int, required=True, help='Number of active learning iterations')
    parser.add_argument('--sampling_strategy', type=str, required=True, help='Active learning sampling strategy')
    parser.add_argument('--n_query', type=int, required=True, help='Number of samples to query per iteration')
    parser.add_argument('--main_task_samples', type=int, required=True, help='Number of main task samples to use')
    parser.add_argument('--main_task', type=int, required=True, help='Index of the task to be used')
    parser.add_argument('--use_all_tasks_to_split', type=bool, required=True, help='Whether to use all tasks in splitting the dataset')
    args = parser.parse_args()
    return vars(args)

def prepare_data(config):
    """Data preparation and split function."""
    data = pd.read_csv(config["target_file"])
    all_tasks = data.loc[:, "NR-AR":"SR-p53"].columns.tolist()
    config["all_tasks"] = all_tasks
    config["main_task"] = [all_tasks[config['main_task']]]
    config["all_tasks"] = all_tasks

    config["aux_task"] = None #["SR-MMP"]
    config["main_task_index"] = 0
    config["aux_task_index"] = None #np.arange(1, len(config["aux_task"])+1)
    
    np.random.seed(config["seed"])
    train_set, test_set = scafoldsplit_train_test(config, all_tasks=config["use_all_tasks_to_split"])
    train_set = drop_unwanted_tasks(train_set, config)
    test_set = drop_unwanted_tasks(test_set, config)
    
    initial_set, train_set = get_initial_set_with_main_and_aux_samples(train_set, config)
    random_stratified_splitter = dc.splits.RandomStratifiedSplitter()
    pool_set, val_set = random_stratified_splitter.train_test_split(train_set, frac_train=0.85, seed=config["seed"])
    
    return convert_to_dataframe(initial_set, config["selected_tasks"]), \
           convert_to_dataframe(val_set, config["selected_tasks"]), \
           convert_to_dataframe(pool_set, config["selected_tasks"]), \
           convert_to_dataframe(test_set, config["selected_tasks"])

def run_active_learning(config, initial_set, pool_set, val_set, test_set):
    """Run active learning iterations."""
    for iteration in range(config["num_iterations"]):
        seed_everything(seed=config["seed"])
        config["iteration"] = iteration
        
        try:
            query_set, updated_training_set, updated_pool_set = active_learning_loop(config, initial_set, pool_set, val_set, test_set)
        except FileNotFoundError:
            print(f"Skipping iteration {iteration}, file not found.")

def main():
    config = get_config_from_args()
    initial_set, val_set, pool_set, test_set = prepare_data(config)
    
    # Active Learning Loop
    run_active_learning(config, initial_set, pool_set, val_set, test_set)

if __name__ == '__main__':
    main()
