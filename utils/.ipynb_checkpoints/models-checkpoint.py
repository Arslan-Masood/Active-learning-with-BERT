import numpy as np
import pandas as pd
from typing import List, Union

import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, auc, matthews_corrcoef, roc_auc_score

from utils.model_utils import NoamLR
# Import necessary modules from chemprop
#from chemprop.models.mpn import MPN
#from chemprop.models.ffn import build_ffn
#from chemprop.nn_utils import initialize_weights

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score,f1_score
from utils.utils import compute_ece
from scipy.special import expit

####################################################
# Chemprop Model
# ####################################################
class Custom_Chemprop(pl.LightningModule):
    def __init__(self, args):
        super(Custom_Chemprop, self).__init__()

        self.scheduler_type = args.scheduler_type
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.steps_per_epoch = args.steps_per_epoch
        self.init_lr = args.init_lr
        self.max_lr = args.max_lr
        self.final_lr = args.final_lr
        self.weight_decay = args.weight_decay
        self.target_weights = args.target_weights
        self.num_of_tasks = args.num_of_tasks

        self.is_atom_bond_targets = args.is_atom_bond_targets
        self.loss_function = args.loss_function
        self.missing_label = args.missing_label_representation
        self.compute_metrics_during_training  = args.compute_metrics_during_training
        self.compute_metric_after_n_epochs = args.compute_metric_after_n_epochs

        # Should we use target weights ?
        if args.use_target_weights:
            if args.data_set == "SIDER":
                complete_data = pd.read_csv(args.data_path + 'SIDER_complete.csv')
                complete_data = complete_data.loc[:,"Hepatobiliary disorders":"Injury, poisoning and procedural complications"]
            else:
                raise ValueError('Provided data_set')
            target_weights = (complete_data == 0).sum() / (complete_data == 1).sum()
            args.target_weights = target_weights.values
            # normalize target weights (Coming from Chemprop)
            avg_weight = sum(args.target_weights)/len(args.target_weights)
            self.target_weights = [w/avg_weight for w in args.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError('Provided target weights must be non-negative.')
            self.target_weights = torch.tensor(self.target_weights, device=args.device).unsqueeze(0)  # shape(1,tasks)
        else:
            self.target_weights = torch.ones(args.num_of_tasks, device=args.device).unsqueeze(0)

        self.loss_fn =  nn.BCEWithLogitsLoss(reduction="none") 
        self.encoder = MPN(args)  # Adjust parameters accordingly
        self.readout = self.readout = build_ffn(
                                        first_linear_dim = args.hidden_size,
                                        hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                                        num_layers=args.ffn_num_layers,
                                        output_size=args.num_of_tasks,
                                        dropout=args.dropout,
                                        activation=args.activation                                    )
        initialize_weights(self)
        

    def forward(self, 
                smiles):
        output = self.encoder(smiles)
        output = self.readout(output)
        return output

    def _shared_step(self, batch, batch_idx):
        device = torch.device("cuda")

        # get batch
        smiles, targets = batch
        smiles = [[SMILES] for SMILES in smiles]
        preds = self(smiles)

        targets = torch.nan_to_num(targets, nan = -1)
        mask = (targets != -1).float()
            
        BCE_loss = self.loss_fn(preds, targets) * mask * self.target_weights
        regularization_loss = torch.tensor([0.0], device = device)
        return BCE_loss, regularization_loss
    
    def configure_optimizers(self):
        # build optimizer
        params = [{"params": self.parameters(), "lr": self.init_lr, "weight_decay": self.weight_decay}]
        self.optimizer = torch.optim.Adam(params)
        # build LR
        self.scheduler = NoamLR(
                                optimizer=self.optimizer,
                                warmup_epochs=self.warmup_epochs,
                                total_epochs= self.epochs,
                                steps_per_epoch= self.steps_per_epoch,
                                init_lr=self.init_lr,
                                max_lr=self.max_lr,
                                final_lr=self.final_lr,
                            )
        
        return {
           'optimizer':  self.optimizer,
           'lr_scheduler':  {"scheduler": self.scheduler,
                             "interval": "step",
                             "frequency": 1,
                             "name": "learning_rate"}
        }
    
    def training_step(self, batch, batch_idx):
        BCE_loss, regularization_loss = self._shared_step(batch,batch_idx)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('train_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('learning_rate_step_end', current_lr, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        return total_loss
    
    def validation_step(self, batch, batch_idx):

        BCE_loss, regularization_loss = self._shared_step(batch, batch_idx)
        total_loss = BCE_loss.mean() + regularization_loss
        self.log('val_BCE_loss', BCE_loss.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_reg_loss', regularization_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size = self.batch_size)

    def on_train_epoch_end(self):
        # Log the learning rate at the end of each epoch
        lr = self.optimizer.param_groups[0]['lr']
        self.log('learning_rate_epoch_end', lr,on_step=False, on_epoch=True)
        
        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.train_dataloader)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision']
            
            for i, score in enumerate(score_list):
                self.log(f'train_{metric[i]}', score, prog_bar=True , on_epoch=True)

    def on_validation_epoch_end(self):
        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.val_dataloaders)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision']
            
            for i, score in enumerate(score_list):
                self.log(f'val_{metric[i]}', score, prog_bar=True , on_epoch=True)
           
    def compute_metrics(self, dataloader): 
        device = torch.device("cuda") 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:

                smiles, batch_targets = batch
                smiles = [[SMILES] for SMILES in smiles]

                batch_preds = self(smiles)
                batch_preds = batch_preds.tolist()

                preds.extend(batch_preds)
                targets.extend(batch_targets.cpu().detach().tolist())

            targets = np.array(targets).reshape(-1,self.num_of_tasks)
            preds = np.array(preds).reshape(-1,self.num_of_tasks)

            if self.missing_label == 'nan':
               mask = ~np.isnan(targets)

            roc_score, blc_acc, sensitivity, specificity, AUPR, f1_score, average_precision = [],[],[],[],[],[],[]
            for i in range(self.num_of_tasks):
                
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
                try:
                    # ROC_AUC
                    fpr, tpr, th = roc_curve(valid_targets, valid_preds)
                    roc_score.append(auc(fpr, tpr))

                    # Balanced accuracy
                    balanced_accuracy = (tpr + (1 - fpr)) / 2
                    blc_acc.append(np.max(balanced_accuracy))

                    # sensitivity, specificity
                    optimal_threshold_index = np.argmax(balanced_accuracy)
                    optimal_threshold = th[optimal_threshold_index]
                    sensitivity.append(tpr[optimal_threshold_index])
                    specificity.append(1 - fpr[optimal_threshold_index])

                    # AUPR, F1
                    precision, recall, thresholds = precision_recall_curve(valid_targets, valid_preds)
                    AUPR.append(auc(recall, precision))
                    f1 = [f1_score(valid_targets, self.prob_to_labels(valid_preds, t)) for t in self.thresholds]
                    f1_score.append(np.nanmax(f1))
                    average_precision.append(average_precision_score(valid_targets, valid_preds))
                    
                except:
                    roc_score.append(np.nan)
                    AUPR.append(np.nan)
                    average_precision.append(np.nan)
                    #print('Performance metric is null')
                
        self.train()
        return np.nanmean(roc_score), np.nanmean(blc_acc), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(AUPR), np.nanmean(f1_score), np.nanmean(average_precision)

    
    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')

####################################################
# focal loss from pytorch
# ####################################################
def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        BCE_loss: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Sourcecode: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss

####################################################
# focal loss Custom implementation
# ####################################################
class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weight):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.w_p = pos_weight


    def forward(self,y_pred, y_true):
        """
        Focal Loss function for binary classification.

        Arguments:
        y_true -- true binary labels (0 or 1), torch.Tensor
        y_pred -- predicted probabilities for the positive class, torch.Tensor

        Returns:
        Focal Loss
        """
        # Compute class weight
        p = torch.sigmoid(y_pred)

        # Compute focal loss for positive and negative examples
        focal_loss_pos = - self.w_p * (1 - p) ** self.gamma * y_true * torch.log(p.clamp(min=1e-8))
        focal_loss_pos_neg = - p ** self.gamma * (1 - y_true) * torch.log((1 - p).clamp(min=1e-8))

        return focal_loss_pos + focal_loss_pos_neg

####################################################
# Residual block
# ##################################################
class Hidden_block(nn.Module):
    def __init__(self,input_dim, hidden_dim, BatchNorm1d, dropout_p, use_skip_connection):
        super(Hidden_block, self).__init__()
        self.use_batch_norm = BatchNorm1d
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.use_skip_connection = use_skip_connection

        if self.use_batch_norm:
            self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x1):
        x2 = self.layer1(x1)

        if self.use_batch_norm:
            x2 = self.batchnorm1(x2) 

        if self.use_skip_connection:
            x2 = x2 + x1             # Add skip connection
            
        x_out = torch.relu(x2)       # apply activation after addition
        x_out = self.dropout(x_out)
        return x_out
    
####################################################
# Vanilla_MLP_classifier
####################################################    
class Vanilla_MLP_classifier(pl.LightningModule):
    def __init__(self, config):
        super(Vanilla_MLP_classifier, self).__init__()

        self.train_step_pos_loss = []
        self.train_step_neg_loss = []
        self.val_step_pos_loss = []
        self.val_step_neg_loss = []

        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.depth = int(config['depth'])
        self.num_of_tasks = config['num_of_tasks']
        self.BatchNorm1d = config['BatchNorm1d']
        self.dropout_p = config['dropout_p']
        self.use_skip_connection = config['use_skip_connection']
        self.loss_type = config['loss_type']
        self.optim = config['optim']
        self.lr = config['lr']
        self.lr_schedulers = config["lr_schedulers"]
        self.epochs = config["epochs"]
        self.compute_metric_after_n_epochs = config["compute_metric_after_n_epochs"]

        self.l2_lambda = config['l2_lambda']
        self.optm_l2_lambda = config['optm_l2_lambda']
        self.batch_size = config['batch_size']
        self.missing = config["missing"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.gpu_status = config["gpu"]

        self.thresholds = np.linspace(0,1,20)

        # pos weights
        pos_weights = pd.read_csv(config["pos_weights"])
        if config["num_of_tasks"] == 1:
            pos_weights = pos_weights.set_index("Targets").reindex([config["selected_tasks"][0]]).weights.values
        else:
            pos_weights = pos_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
        
        pos_weights = (config["alpha"] * pos_weights) + (1 - config["alpha"])*1
        self.pos_weights = torch.tensor(pos_weights, device = config["device"])
        alpha_null = torch.isnan(self.pos_weights).any()
        assert not alpha_null, "There are null values in the pos_weight tensor"

        # class weights
        if config['num_of_tasks'] > 1:
            class_weights = pd.read_csv(config["class_weights"])
            class_weights = class_weights.set_index("Targets").reindex(config["selected_tasks"]).weights.values
            class_weights = (config["beta"] * class_weights) + (1 - config["beta"])*1
            self.class_weights = torch.tensor(class_weights, device = config["device"])
        else:
            self.class_weights = torch.tensor([1.0], device = config["device"])
        beta_null = torch.isnan(self.class_weights).any()
        assert not beta_null, "There are null values in the class_weight tensor"

        # train_weighted loss, validation no weights
        self.weighted_creterien =  nn.BCEWithLogitsLoss(reduction="none", 
                                                        pos_weight= self.pos_weights,
                                                        weight= self.class_weights)
        
        self.non_weighted_creterian =  nn.BCEWithLogitsLoss(reduction="none")
        self.FL = FocalLoss(gamma=config['gamma'], pos_weight= self.pos_weights)

        # Model architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.Hidden_block = nn.ModuleList([Hidden_block(self.hidden_dim, 
                                                        self.hidden_dim, 
                                                        self.BatchNorm1d, 
                                                        self.dropout_p,
                                                        self.use_skip_connection
                                                        ) for _ in range(self.depth)])
        self.output_layer = nn.Linear(self.hidden_dim, self.num_of_tasks)
        
        # dropout and Batchnorm for first layer output
        self.dropout = nn.Dropout(self.dropout_p)
        if self.BatchNorm1d:
            self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, x_input):
        x1 = self.input_layer(x_input)
        if self.BatchNorm1d:
            x1 = self.batchnorm1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout(x1)
        
        for block in self.Hidden_block:
            x_n = block(x1)  # Apply each Hidden block
        x_output = self.output_layer(x_n)
        return x_output
    
    def configure_optimizers(self):
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.lr)
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              weight_decay = self.optm_l2_lambda,
                                             lr=self.lr)
        
        if self.lr_schedulers == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max = 10, 
                                                                        eta_min=1e-6) 
            return {"optimizer": self.optimizer, 
                    "lr_scheduler": self.scheduler}
        
        if self.lr_schedulers == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        verbose=True,
                                                                        patience=15,
                                                                        min_lr=1e-6,
                                                                        mode = 'min')
            return {
            'optimizer':  self.optimizer,
            'lr_scheduler':  self.scheduler, # Changed scheduler to lr_scheduler
            'monitor': 'val_BCE_loss'
            }
    def l2_regularization(self):
        if self.gpu_status == None:
            device = "cpu"
        else:
            device = torch.device("cuda") 
        l2_reg = torch.tensor(0., requires_grad=True, device=device)

        # Apply only on weights, exclude bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return l2_reg
    
    def _compute_loss(self, y, y_hat):
        #if self.num_of_tasks == 1:
        #    y = y.unsqueeze(1)
        # compute losses, wiht masking
        if self.missing == 'nan':
            y = torch.nan_to_num(y, nan = -1)
        
        # masks
        valid_label_mask = (y != -1).float()
        pos_label_mask = (y == 1)
        negative_label_mask = (y == 0)

        if self.loss_type == "BCE":
            weighted_loss = self.weighted_creterien(y_hat, y) * valid_label_mask
        if self.loss_type == "Focal_loss":
            weighted_loss = self.FL(y_hat, y)* valid_label_mask
        Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask

        if self.loss_type == 'Focal_loss_v2':
            Non_weighted_loss = self.non_weighted_creterian(y_hat, y) * valid_label_mask
            weighted_loss = sigmoid_focal_loss(inputs = y_hat,
                                                targets = y,
                                                BCE_loss = Non_weighted_loss,
                                                alpha = self.alpha,
                                                gamma = self.gamma)
            weighted_loss = weighted_loss * valid_label_mask
        
        # Non_weighted_loss, positive negative loss
       
        pos_loss = Non_weighted_loss * pos_label_mask
        neg_loss = Non_weighted_loss * negative_label_mask
        pos_loss = pos_loss.sum() / pos_label_mask.sum()
        neg_loss = neg_loss.sum() / negative_label_mask.sum()
    

        # compute mean loss
        Non_weighted_loss = Non_weighted_loss.sum() / valid_label_mask.sum()
        weighted_loss = weighted_loss.sum() / valid_label_mask.sum()

        l2_reg_loss = self.l2_regularization()
        l2_reg_loss = self.l2_lambda*l2_reg_loss
        total_loss = weighted_loss + l2_reg_loss

        return total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss

    def training_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.train_step_pos_loss.append(pos_loss.item())
        self.train_step_neg_loss.append(neg_loss.item())

        self.log('train_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('learning_rate_step_end', current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # compute forward pass
        x, y = batch
        y_hat = self(x)

        # compute loss
        total_loss, weighted_loss, Non_weighted_loss,l2_reg_loss, pos_loss, neg_loss = self._compute_loss(y, y_hat)  
        self.val_step_pos_loss.append(pos_loss.item())
        self.val_step_neg_loss.append(neg_loss.item())

        self.log('val_BCE_weighted', weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_BCE_non_weighted', Non_weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_l2_reg_loss', l2_reg_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):

        pos_loss = torch.tensor(self.train_step_pos_loss)
        neg_loss = torch.tensor(self.train_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())

        self.log('train_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('train_gm_loss', geometric_mean,on_step=False, on_epoch=True)
    
        current_lr = self.optimizer.param_groups[0]['lr']
        self.log('learning_rate_epoch_end', current_lr, prog_bar=True, on_step=False, on_epoch=True)
        
        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.train_dataloader)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'train_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.train_step_pos_loss.clear()
        self.train_step_neg_loss.clear()


    def on_validation_epoch_end(self):

        pos_loss = torch.tensor(self.val_step_pos_loss)
        neg_loss = torch.tensor(self.val_step_neg_loss)
        geometric_mean = torch.sqrt(pos_loss.nanmean() * neg_loss.nanmean())
        
        self.log('val_BCE_pos', pos_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_BCE_neg', neg_loss.nanmean(),on_step=False, on_epoch=True)
        self.log('val_gm_loss', geometric_mean,on_step=False, on_epoch=True)


        if (self.current_epoch % self.compute_metric_after_n_epochs == 0) or (self.current_epoch == 0) or (self.current_epoch == self.epochs -1):
            score_list =  self.compute_metrics(self.trainer.val_dataloaders)
            metric = ['roc_score', 'blc_acc', 'sensitivity', 'specificity', 'AUPR', 'f1_score', 'average_precision','ECE_score','ACE_score']
            
            for i, score in enumerate(score_list):
                self.log(f'val_{metric[i]}', score, prog_bar=True , on_epoch=True)

        self.val_step_pos_loss.clear()
        self.val_step_neg_loss.clear()
           
    def compute_metrics(self, dataloader): 
        if self.gpu_status == None:
            device = "cpu"
        else:
            device = torch.device("cuda") 
        self.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:

                batch_x,batch_targets = batch
                batch_preds = self(batch_x.to(device))

                preds.extend(batch_preds.cpu().detach().tolist())
                targets.extend(batch_targets.cpu().detach().tolist())

            targets = np.array(targets).reshape(-1,self.num_of_tasks)
            preds = np.array(preds).reshape(-1,self.num_of_tasks)

            if self.missing == 'nan':
               mask = ~np.isnan(targets)

            roc_score, blc_acc, sensitivity, specificity, AUPR, f1, average_precision = [],[],[],[],[],[],[]
            ECE_score, ACE_score = [],[]

            n_bins = 10

            for i in range(self.num_of_tasks):
                
                # get valid targets, and convert logits to prob
                valid_targets = targets[:,i][mask[:,i]]
                valid_preds = expit(preds[:,i][mask[:,i]])
                ECE= compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = True)
                ACE = compute_ece(valid_targets, valid_preds, n_bins=10, equal_intervals = False)
                ECE_score.append(ECE)
                ACE_score.append(ACE)

                try:
                    # ROC_AUC
                    fpr, tpr, th = roc_curve(valid_targets, valid_preds)
                    roc_score.append(auc(fpr, tpr))

                    # Balanced accuracy
                    balanced_accuracy = (tpr + (1 - fpr)) / 2
                    blc_acc.append(np.max(balanced_accuracy))

                    # sensitivity, specificity
                    optimal_threshold_index = np.argmax(balanced_accuracy)
                    optimal_threshold = th[optimal_threshold_index]
                    sensitivity.append(tpr[optimal_threshold_index])
                    specificity.append(1 - fpr[optimal_threshold_index])

                    # AUPR, F1
                    precision, recall, thresholds = precision_recall_curve(valid_targets, valid_preds)
                    AUPR.append(auc(recall, precision))
                    f1_sc = f1_score(valid_targets, self.prob_to_labels(valid_preds, optimal_threshold))
                    f1.append(f1_sc)
                    average_precision.append(average_precision_score(valid_targets, valid_preds))
                    
                except:
                    roc_score.append(np.nan)
                    AUPR.append(np.nan)
                    average_precision.append(np.nan)
                    #print('Performance metric is null')
                
        self.train()
        return np.nanmean(roc_score), np.nanmean(blc_acc), np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(AUPR), np.nanmean(f1), np.nanmean(average_precision),np.nanmean(ECE_score),np.nanmean(ACE_score)

    
    def prob_to_labels(self, pred, threshold):
	    return (pred >= threshold).astype('int')
