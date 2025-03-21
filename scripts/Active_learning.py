import os
import argparse
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import gc

import wandb
os.environ["WANDB_SILENT"] = "true"
wandb.login(key = "27edf9c66b032c03f72d30e923276b93aa736429")

import hjson  # Use HJSON instead of json
import torch
import torch.multiprocessing as mp
import deepchem as dc

import sys
sys.path.insert(1, '/scratch/work/masooda1/active_learning')
from pytorch_lightning import seed_everything
from utils.data_utils import (read_train_test_files, scafoldsplit_train_test, convert_to_dataframe, 
                               drop_unwanted_tasks, get_initial_set_with_main_and_aux_samples)

from utils.data_utils import convert_dataframe_to_dataloader
from utils.model_utils import get_pred_with_uncertainities
from utils.utils import wandb_init_model, compute_binary_classification_metrics_MT, active_learning_loop


class ConfigManager:
    def __init__(self, json_file, seed=None):
        """Load configuration from a JSON file and set the seed."""
        self.config = self.get_config_from_args(json_file)
        self.config["seed"] = seed  # Add seed to config
        self.prepare_directories()

    def get_config_from_args(self, json_file):
        """Load configuration from a JSON file."""
        with open(json_file, 'r') as f:
            config = hjson.load(f)
        return config

    def prepare_directories(self):
        """Build directory paths based on config and create the directories if they don't exist."""
        t_names = os.path.join(self.config["metadata_dir"], self.config["sampling_strategy"], self.config["main_task"])
        self.config["query_set_dir"] = os.path.join(t_names, "query_set")
        self.config["result_dir"] = os.path.join(t_names, "Results")
        self.config["model_weights_dir"] = os.path.join(t_names, "model_weights")

        # Select device
        if self.config["gpu"] == [0]:
            self.config["device"]  = torch.device("cuda:0")
        elif self.config["gpu"] == [1]:
            self.config["device"]  = torch.device("cuda:1")
        elif self.config["gpu"] == [2]:
            self.config["device"]  = torch.device("cuda:2")
        elif self.config["gpu"] == [3]:
            self.config["device"]  = torch.device("cuda:3")
        else:
            self.config["device"]  = "cpu"
        
        print(self.config["device"])

        # Create directories
        directories = [self.config["query_set_dir"], self.config["result_dir"], self.config["model_weights_dir"]]
        try:
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                print(f"Directory created or already exists: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

    def prepare_data(self):
        """Data preparation and split function."""
        
        print("Step 1: Reading input file")
        self.config["selected_tasks"] = self.config["main_task"]
        
        # Ensure selected_tasks is a list
        if not isinstance(self.config["selected_tasks"], list):
            self.config["selected_tasks"] = [self.config["selected_tasks"]]
        self.config["num_of_tasks"] = len(self.config["selected_tasks"])

        print("Step 2: Getting train test data")
        np.random.seed(self.config["seed"])
        
        # Choose between reading existing split files or performing scaffold split
        if self.config.get("train_test_split_exists", False):
            print("Using existing train/test split files")
            train_set, test_set = read_train_test_files(self.config, all_tasks=self.config["use_all_tasks_to_split"])
        else:
            print("Performing scaffold split")
            train_set, test_set = scafoldsplit_train_test(self.config, all_tasks=self.config["use_all_tasks_to_split"])
        
        if train_set.y.shape[1] > 1:
            train_set = drop_unwanted_tasks(train_set, self.config)
            test_set = drop_unwanted_tasks(test_set, self.config)

        print("Step 3: Creating initial and pool set")
        initial_set, train_set = get_initial_set_with_main_and_aux_samples(train_set, self.config)
        random_stratified_splitter = dc.splits.RandomStratifiedSplitter()
        pool_set, val_set = random_stratified_splitter.train_test_split(train_set, frac_train=0.85, seed=self.config["seed"])

        selected_tasks = self.config["selected_tasks"] if isinstance(self.config["selected_tasks"], list) else [self.config["selected_tasks"]]
        return (convert_to_dataframe(initial_set, selected_tasks), 
                convert_to_dataframe(val_set, selected_tasks), 
                convert_to_dataframe(pool_set, selected_tasks), 
                convert_to_dataframe(test_set, selected_tasks))

    def save_initial_set(self, initial_set):
        file_path = os.path.join(self.config["query_set_dir"], "initial_set.csv")
        initial_set.to_csv(file_path, index=False)



class ActiveLearningRunner:
    def __init__(self, config_manager, initial_set, pool_set, val_set, test_set):
        self.config = config_manager.config
        self.initial_set = initial_set
        self.pool_set = pool_set
        self.val_set = val_set
        self.test_set = test_set
        
        # Initialize file paths
        base_path = os.path.join(
            self.config["metadata_dir"], 
            self.config["sampling_strategy"], 
            self.config["main_task"]
        )
        
        # Query history file
        self.query_history_file = os.path.join(
            base_path, 
            "query_set",
            f'query_history_seed{self.config["seed"]}.csv'
        )
        
        # Metrics history file
        self.metrics_history_file = os.path.join(
            base_path,
            "Results",
            f'metrics_history_seed{self.config["seed"]}.csv'
        )
        
        # Initialize query history file
        if not os.path.exists(self.query_history_file):
            columns = list(pool_set.columns) + ['iteration']
            pd.DataFrame(columns=columns).to_csv(self.query_history_file, index=False)
            
        # Initialize metrics history file with your specific metrics
        if not os.path.exists(self.metrics_history_file):
            columns = ['iteration', 'Tasks', 'balanced_acc', 'f1_score', 'specificity', 
                      'sensitivity', 'roc_auc', 'AUPR', 'average_precision', 'ECE', 'ACE']
            pd.DataFrame(columns=columns).to_csv(self.metrics_history_file, index=False)

    def generate_model_name(self):
        """Generate model name string."""
        return (rf'itteration_{self.config["iteration"]}_s{self.config["seed"]}_alpha_{self.config["alpha"]}_'
                rf'gamma_{self.config["gamma"]}_loss_type_{self.config["loss_type"]}_λ{self.config["optm_l2_lambda"]}')

    def model_exists(self):
        """Check if model exists by looking for query points in history file."""
        if not os.path.exists(self.query_history_file):
            return False
        
        try:
            # Read the query history
            query_history = pd.read_csv(self.query_history_file)
            # Check if this iteration's queries exist
            return len(query_history[query_history['iteration'] == self.config["iteration"]]) > 0
        except (pd.errors.EmptyDataError, pd.errors.ParserError, PermissionError) as e:
            print(f"Error reading query history file: {e}")
            return False

    def get_dataloaders(self):
        """Convert dataframes to dataloaders."""
        train_dl = convert_dataframe_to_dataloader(self.initial_set, self.config, shuffle=True, drop_last=True)
        val_dl = convert_dataframe_to_dataloader(self.val_set, self.config, shuffle=False, drop_last=False)
        pool_dl = convert_dataframe_to_dataloader(self.pool_set, self.config, shuffle=False, drop_last=False)
        test_dl = convert_dataframe_to_dataloader(self.test_set, self.config, shuffle=False, drop_last=False)
        return train_dl, val_dl, pool_dl, test_dl

    def train_model(self, train_dl, val_dl):
        """Train the model and initialize wandb."""
        from utils.models import Vanilla_MLP_classifier
        self.config["training_steps"] = len(train_dl)
        trained_model, run, trainer = wandb_init_model(
            model=Vanilla_MLP_classifier, 
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            config=self.config, 
            model_type='MLP'
        )
        # Log the best validation metric
        best_model_score = trainer.checkpoint_callback.best_model_score
        if best_model_score is not None:
            print(f"Best validation score: {best_model_score:.4f}")
        wandb.finish()
        return trained_model

    def evaluate_model(self, trained_model, test_dl):
        """Evaluate the trained model and return predictions."""
        trained_model = trained_model.eval()  # Set model to evaluation mode
        targets, pred_mean, pred_var, all_pred = get_pred_with_uncertainities(
            test_dl, trained_model,
            n_classes=self.config["num_of_tasks"],
            cal_uncert=False,
            num_forward_passes=1
        )
        return targets, pred_mean

    def save_metrics(self, targets, pred_mean, iteration):
        """Compute and save metrics."""
        # Compute metrics
        metrics = compute_binary_classification_metrics_MT(targets, pred_mean, missing='nan')
        metrics = metrics.append(metrics.mean(), ignore_index=True)
        metrics.insert(0, 'Tasks', self.config["selected_tasks"] + ['mean'])
        metrics.insert(0, 'iteration', iteration)
        
        # Append to metrics history file
        if os.path.getsize(self.metrics_history_file) == 0:
            metrics.to_csv(self.metrics_history_file, index=False)
        else:
            # Check if headers match
            existing_headers = pd.read_csv(self.metrics_history_file, nrows=0).columns.tolist()
            if existing_headers != metrics.columns.tolist():
                print(f"Warning: Metrics headers mismatch. Expected: {existing_headers}, Got: {metrics.columns.tolist()}")
            # Append without headers
            metrics.to_csv(self.metrics_history_file, mode='a', header=False, index=False)
        
        # Print current metrics
        print(f"Iteration {iteration} metrics:")
        print(metrics.tail(1))  # Print only the mean row

    def acquire_sample(self, trained_model, pool_dl, test_dl):
        """Run an iteration of active learning and update the training and pool sets."""
        if self.config["sampling_strategy"] == "EPIG_MT":
            query_set, updated_training_set, updated_pool_set = active_learning_loop(
                trained_model=trained_model, 
                pool_dataloader=pool_dl, 
                initial_set=self.initial_set, 
                pool_set=self.pool_set, 
                config=self.config,
                query_set=None,
                test_dataloader=test_dl,
                test_set=self.test_set
            )
        else: 
            query_set, updated_training_set, updated_pool_set = active_learning_loop(
                trained_model=trained_model, 
                pool_dataloader=pool_dl, 
                initial_set=self.initial_set, 
                pool_set=self.pool_set, 
                config=self.config
            )

        # Add iteration number to query set
        query_set['iteration'] = self.config["iteration"]
        
        # Append the query set to the history file
        if os.path.getsize(self.query_history_file) == 0:
            # If file is empty, write with headers
            query_set.to_csv(self.query_history_file, index=False)
        else:
            # Check if headers match
            existing_headers = pd.read_csv(self.query_history_file, nrows=0).columns.tolist()
            if existing_headers != query_set.columns.tolist():
                print(f"Warning: Headers mismatch. Expected: {existing_headers}, Got: {query_set.columns.tolist()}")
            # Append without headers
            query_set.to_csv(self.query_history_file, mode='a', header=False, index=False)

        # Update the initial and pool sets
        self.initial_set = updated_training_set.copy()
        self.pool_set = updated_pool_set.copy()

    def update_sets(self, iteration):
        """Update training and pool sets if model exists."""
        # Read the query history
        query_history = pd.read_csv(self.query_history_file)
        # Get query set for this iteration
        query_set = query_history[query_history['iteration'] == iteration].drop('iteration', axis=1)
        
        _, updated_training_set, updated_pool_set = active_learning_loop(
            trained_model=None, 
            pool_dataloader=None, 
            initial_set=self.initial_set, 
            pool_set=self.pool_set, 
            config=self.config, 
            query_set=query_set
        )
        self.initial_set = updated_training_set.copy()
        self.pool_set = updated_pool_set.copy()

    def run(self):
        """Run active learning iterations."""
        for iteration in range(self.config["num_iterations"]):
            seed_everything(seed=self.config["seed"])
            self.config["iteration"] = iteration
            self.config["model_name"] = self.generate_model_name()

            # Add model checkpoint name format to config
            self.config["checkpoint_name"] = f'model-iter{iteration}-{self.config["main_task"]}-{{epoch:02d}}-{{val_BCE_non_weighted:.4f}}' 

            if self.model_exists():
                self.update_sets(iteration)
            else:
                # Get the data loaders
                train_dl, val_dl, pool_dl, test_dl = self.get_dataloaders()

                # Train the model
                trained_model = self.train_model(train_dl, val_dl)
  
                # Evaluate the model
                targets, pred_mean = self.evaluate_model(trained_model, test_dl)

                # Save the metrics
                self.save_metrics(targets, pred_mean, iteration)

                # Run active learning iteration and update sets
                self.acquire_sample(trained_model, pool_dl, test_dl)

                # Clean up memory
                del train_dl, val_dl, test_dl, pool_dl, trained_model
                torch.cuda.empty_cache()
                gc.collect()

                active_procs = mp.active_children()
                # Loop through the list of active child processes and terminate them
                for proc in active_procs:
                    proc.terminate()

def main():
    # Get the JSON file name and seed from command line arguments
    parser = argparse.ArgumentParser(description='Active Learning Config')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON configuration file')
    parser.add_argument('--seed', type=int, required=True, help='Seed for random number generation')
    args = parser.parse_args()

    print("Step 1: Config manager initialized")
    config_manager = ConfigManager(args.config_file, seed=args.seed)
    print("Step 2: Preparing data")
    initial_set, val_set, pool_set, test_set = config_manager.prepare_data()
    config_manager.save_initial_set(initial_set)
    
    # Active Learning Loop
    print("Step 3: Active learning runner initialized")
    active_learning_runner = ActiveLearningRunner(config_manager, initial_set, pool_set, val_set, test_set)
    active_learning_runner.run()


if __name__ == '__main__':
    main()
