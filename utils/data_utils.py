import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.data import NumpyDataset
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

import torch
########################################################
# scafoldsplit by using all tasks
# Then drop unwanted tasks
# To make sure we have same testset in all cases
#####################################################
def drop_unwanted_tasks(dc_dataset, config):
    task_index = [config["all_tasks"].index(i) for i in config["selected_tasks"]]
    updated_dataset = NumpyDataset(X = dc_dataset.X, y = dc_dataset.y[:,task_index], ids = dc_dataset.ids)
    return updated_dataset

########################################################
# scafoldsplit_train_test
#####################################################
def scafoldsplit_train_test(args, all_tasks = False):
    try:
        config = vars(args)
    except:
        config = args
    np.random.seed(config["seed"])
    
    if config["features_type"] == "FP":
        data = pd.read_csv(config["target_file"])
        BERT_features = pd.read_csv(config["BERT_features_file"])
        ECFP_features = pd.read_csv(config["ECFP_features_file"])

        data = data[data.SMILES.isin(BERT_features["SMILES"])]
        data = data.drop_duplicates(subset = "SMILES")
        data = data.reset_index(drop = True)

        ECFP_features = ECFP_features[ECFP_features.SMILES.isin(data["SMILES"])]
        ECFP_features = ECFP_features.drop_duplicates(subset = "SMILES")
        ECFP_features = ECFP_features.reset_index(drop = True)

        if all_tasks:
            y = data.loc[:,config["all_tasks"]].values
        else:
            y = data.loc[:,config["selected_tasks"]].values
        X = ECFP_features.iloc[:,1:].values

    if config["features_type"] == "BERT":
        data = pd.read_csv(config["target_file"])
        BERT_features = pd.read_csv(config["BERT_features_file"])
        data = data[data.SMILES.isin(BERT_features["SMILES"])]
        data = data.drop_duplicates(subset = "SMILES")
        data = data.reset_index(drop = True)

        if all_tasks:
            y = data.loc[:,config["all_tasks"]].values
        else:
            y = data.loc[:,config["selected_tasks"]].values
        X = BERT_features.iloc[:,1:].values

    # split data into train and test
    splitter = dc.splits.ScaffoldSplitter()
    dc_dataset = NumpyDataset(X = X, y = y, ids = data.SMILES)
    train_set, test_set = splitter.train_test_split(dataset = dc_dataset,
                                                    frac_train = config["train_frac"], 
                                                    seed = 42)
    print('train_test_features',train_set.X.shape, test_set.X.shape)
    print('train_test_targets',train_set.y.shape, test_set.y.shape)
    return train_set, test_set

########################################################
# dataloader_for_numpy
#####################################################
class dataloader_for_numpy(Dataset):

    def __init__(self, X, y, x_type = 'SMILES'):
        
        if x_type == 'SMILES':
            self.x = X.tolist()
            self.n_samples = len(self.x)
        else:
            self.x = torch.tensor(X, dtype=torch.float32)
            self.n_samples = self.x.shape[0]

        self.y = torch.tensor(y, dtype=torch.float32)
        

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def get_stratified_folds(X, y, ids, num_of_folds, config):

    if config["num_of_tasks"] > 1:
        mskf = MultilabelStratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=config["seed"])
    else:
        mskf = StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=config["seed"])
    fold_data = []
    for fold_idx, (train_index, val_index) in enumerate(mskf.split(X, np.nan_to_num(y, nan=0))):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        ids_train, ids_val = ids[train_index], ids[val_index]

        if config["num_of_tasks"] > 1:
            class_sum = min(np.nansum(y_val, axis = 0))
            if np.min(class_sum) < 1:
                raise ValueError("Error: No active compound in certain class")
        # Store the data for the current fold
        fold_data.append({
            'fold_idx': fold_idx,
            'train': {'X': X_train, 'y': y_train, 'ids': ids_train},
            'val': {'X': X_val, 'y': y_val,'ids': ids_val}
        })
    return fold_data
   
def convert_to_dataframe(dc_datset, tasks_name):
    dataset = pd.DataFrame(dc_datset.y)
    dataset.columns = tasks_name
    dataset.insert(0, "SMILES", dc_datset.ids)
    return dataset

def get_features(dataset, config):
    SMILES = pd.DataFrame({'SMILES': dataset.SMILES})
    if config["features_type"] == "BERT":
        features = pd.read_csv(config["BERT_features_file"])
    if config["features_type"] == "FP":
        features = pd.read_csv(config["ECFP_features_file"])

    if config["features_type"] == "DILI_dataset":
        seed = config["seed"]
        train_features = pd.read_csv(config["target_dir"] + f"train_DILI_BERT_features_seed_{seed}.csv")
        valid_features = pd.read_csv(config["target_dir"] + f"valid_DILI_BERT_features_seed_{seed}.csv")
        test_features = pd.read_csv(config["target_dir"] + f"test_DILI_BERT_features_seed_{seed}.csv")
        features = pd.concat([train_features,valid_features, test_features]).reset_index(drop = True)

    X = pd.merge(SMILES, features, on = "SMILES", how = "inner")
    X = X.drop(columns= 'SMILES').values
    X = pd.merge(SMILES, features, on = "SMILES", how = "inner")
    X = X.drop(columns= 'SMILES').values
    print("After merging", X.shape)
    return X

def convert_dataframe_to_dataloader(dataframe, config, shuffle = False, drop_last = False):
    
    X = get_features(dataframe, config)
    y = dataframe[config["selected_tasks"]].values
    dataloader = DataLoader(dataloader_for_numpy(X, y, x_type = 'Fingerprints'), 
                                                                batch_size=config["batch_size"],
                                                                pin_memory=False,
                                                                num_workers=1, 
                                                                shuffle = shuffle,
                                                                persistent_workers=False,
                                                                drop_last = drop_last)
    return dataloader

def get_initial_set_with_equal_ratio_of_active_inactive(train_set, args):

    train_active_indices = np.where(np.any(train_set.y == 1, axis=1))[0]
    train_inactive_indices = np.where(np.all(train_set.y == 0, axis=1))[0]

    # Randomly select 50 inactive and 50 active compounds for each task
    np.random.seed(args.seed)  # For reproducibility
    num_samples_per_class = int(args.initial_set_size / 2)
    selected_indices = np.concatenate([
        np.random.choice(train_active_indices, num_samples_per_class, replace=False),
        np.random.choice(train_inactive_indices, num_samples_per_class, replace=False)
    ])
    # Create the validation set
    initial_set = train_set.select(selected_indices)
    trainset_indices = np.setdiff1d(np.arange(len(train_set)), selected_indices)

    updated_train_set = train_set.select(trainset_indices)
    num_active_compounds = np.sum(initial_set.y[:, 0] == 1)
    num_inactive_compounds = np.sum(initial_set.y[:, 1] == 1)
    print("Number of active compounds:", num_active_compounds)
    print("Number of inactive compounds:", num_inactive_compounds)

    return initial_set, updated_train_set

def get_random_query_set(dataset,config):
    np.random.seed(config["seed"])
    non_nan_indices = get_non_na_ind(dataset)
    if non_nan_indices.shape[0] > config["n_query"]:
        selected_set = np.random.choice(non_nan_indices.shape[0], size=config["n_query"], replace=False)
        selected_mol_tasks = non_nan_indices[selected_set]
    else:
        selected_mol_tasks = non_nan_indices
    
    top_guns = pd.DataFrame()
    for mol_ind, task_ind in selected_mol_tasks:
        
        selected_sample = dataset[dataset.SMILES == mol_ind].loc[:, ["SMILES",task_ind]]
        top_guns = pd.concat([top_guns, selected_sample],axis=0)
    
    # Merge rows with SIMILAR SMILES
    top_guns = top_guns.groupby('SMILES').sum(min_count = 1).reset_index()
    return top_guns
   
def get_non_na_ind(dataset):
    dataset = dataset.reset_index(drop = True)
    if "SMILES" in dataset.columns:
        row_indices, col_indices = np.where(dataset.iloc[:,1:].notna().values)
        non_nan_indices = np.column_stack((dataset.SMILES[row_indices], dataset.columns[col_indices + 1]))
    else:
        "SMILES is not the first column, check me out"
    return non_nan_indices

def get_query_set(pool_set, top_indices):

    # get top guns
    selected_mol = pool_set.iloc[np.unique(top_indices[0]),0]

    # diffuse all others ~y_ik i.e replace with nan
    p_set = pool_set.iloc[:, 1:].copy(deep=True)
    mask = np.ones_like(p_set, dtype=bool)
    mask[top_indices] = False
    p_set[mask] = np.nan

    # pull our guys (y_ik)
    selected_tasks = p_set.iloc[np.unique(top_indices[0]), :]
    top_guns = pd.concat([selected_mol, selected_tasks], axis = 1).reset_index(drop = True)
    top_guns = top_guns.reset_index(drop = 2)
    return top_guns

def update_training_set(train_set, query_set):

    train_set = train_set.set_index("SMILES")
    query_set = query_set.set_index("SMILES")
    train_set = train_set.combine_first(query_set)
    train_set = train_set.reset_index()
    return train_set

def remove_queried_index_from_pool_set(pool_set, query_set, config):
    try:
        config = vars(config)
    except:
        pass

    non_nan_indices = get_non_na_ind(query_set)
    updated_poolset = pool_set.copy(deep = True)

    # Replace observed values with nan
    for mol_ind, task_ind in non_nan_indices:
        updated_poolset.loc[updated_poolset['SMILES'] == mol_ind, task_ind] = np.nan
    
    # drop those rows whose all labels has be exhausted
    updated_poolset = updated_poolset.dropna(subset = config["selected_tasks"], how = "all")

    return updated_poolset

def remove_queried_mol_from_pool_set(pool_set, top_indices, aux_task):
    '''
    # Single Task sampling
    '''
    # update pool set
    SMILES = pool_set.iloc[:,0]
    updated_poolset = pool_set.iloc[:, 1:].copy(deep=True)
    mask = np.zeros_like(updated_poolset, dtype=bool)
    mask[top_indices] = True
    updated_poolset[mask] = np.nan
    updated_poolset = pd.concat([SMILES, updated_poolset], axis = 1).reset_index(drop = True)
    updated_poolset.dropna(subset= aux_task, inplace = True)
    updated_poolset = updated_poolset.reset_index(drop = True)
    return updated_poolset

def get_1_task_samples(selected_aux_task_index, config, train_set):

    np.random.seed(config["seed"])
    all_aux_task_index = config["aux_task_index"].copy()
    #selected_aux_task_index = config["aux_task_index"][0]
    remaining_aux_index = list(set(all_aux_task_index) - set([selected_aux_task_index]))

    aux_task_1 = train_set.y[:,selected_aux_task_index]
    active_indices = np.where(aux_task_1 == 1)[0]
    inactive_indices = np.where(aux_task_1 == 0)[0]

    num_samples_per_class = int(config["aux_task_samples"] /2)
    selected_indices = np.concatenate([
            np.random.choice(active_indices, num_samples_per_class, replace=False),
            np.random.choice(inactive_indices, num_samples_per_class, replace=False)
        ])
    task_data = train_set.select(selected_indices)
    # We are selecting only main task, hide aux tasks
    task_data.y[:, [config["main_task_index"]] + remaining_aux_index] = np.nan
    task_data = convert_to_dataframe(task_data, config["selected_tasks"])
    return task_data

def get_initial_set_with_main_and_aux_samples(train_set, config):
    try:
        config = vars(config)
    except:
        pass
    np.random.seed(config["seed"])
    main_task = train_set.y[:,config["main_task_index"]]
    main_task_active_indices = np.where(main_task == 1)[0]
    main_task_inactive_indices = np.where(main_task == 0)[0]

    num_samples_per_class = int(config["main_task_samples"] /2)
    main_selected_indices = np.concatenate([
            np.random.choice(main_task_active_indices, num_samples_per_class, replace=False),
            np.random.choice(main_task_inactive_indices, num_samples_per_class, replace=False)
        ])
    if config["aux_task"] != None:
        main_task_data = train_set.select(main_selected_indices)
        # We are selecting only main task, hide aux tasks
        main_task_data.y[:,config["aux_task_index"]] = np.nan
        initial_set_data = convert_to_dataframe(main_task_data, config["selected_tasks"])

        for selected_aux_task_index in config["aux_task_index"]:

            aux_task_data = get_1_task_samples(selected_aux_task_index, config, train_set)
            # combine both sets
            initial_set_data = update_training_set(initial_set_data, aux_task_data)
    else:
        main_task_data = train_set.select(main_selected_indices)
        initial_set_data = convert_to_dataframe(main_task_data, config["selected_tasks"])

    # Remove inital_set from training set
    train_set = convert_to_dataframe(train_set, config["selected_tasks"])
    train_set = remove_queried_index_from_pool_set(train_set, 
                                    initial_set_data, config)
    
    # convert back to deepchem dataset object
    train_set = dc.data.NumpyDataset(X=train_set.SMILES.values, 
                            y=train_set[config["selected_tasks"]].values, 
                            ids=train_set.SMILES.values)
    
    initial_set_data = dc.data.NumpyDataset(X=initial_set_data.SMILES.values, 
                            y=initial_set_data[config["selected_tasks"]].values, 
                            ids=initial_set_data.SMILES.values)
    
    return initial_set_data, train_set

def convert_df_to_dc_data_object(df, config):
    dc_object = dc.data.NumpyDataset(X= df.SMILES.values, 
                                    y= df[config["all_tasks"]].values, 
                                    ids= df.SMILES.values)
    return dc_object