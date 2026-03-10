#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch

##############################################################################
# 1) Read Data
##############################################################################

def read_data(file_path: str, rep_col_name: str, target_col_names: list) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\t')
    print("Columns in the DataFrame are:", df.columns.tolist())
    columns = [rep_col_name] + target_col_names
    df = df[columns].copy()
    return df

##############################################################################
# 2) Split Data
##############################################################################

def train_val_test_split(df: pd.DataFrame, split_fracs: tuple = (0.85, 0.1, 0.05), random_state: int = 42):
    assert len(split_fracs) == 3, "split_fracs must have length 3."
    assert abs(sum(split_fracs) - 1.0) < 1e-7, "split_fracs must sum to 1."
    train_frac, val_frac, test_frac = split_fracs
    df_train, df_temp = train_test_split(df, test_size=(1.0 - train_frac), random_state=random_state)
    relative_val_frac = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(df_temp, test_size=(1.0 - relative_val_frac), random_state=random_state)
    return df_train, df_val, df_test

##############################################################################
# 3) DataModule
##############################################################################

class MultiTaskDataset(Dataset):
    def __init__(self, df, rep_col_name, target_col_names):
        super().__init__()
        parsed_features = []
        for item in df[rep_col_name]:
            if isinstance(item, str):
                arr = np.array(ast.literal_eval(item), dtype=np.float32)
            else:
                arr = np.array(item, dtype=np.float32)
            parsed_features.append(arr)

        self.features = np.stack(parsed_features, axis=0)
        self.targets = df[target_col_names].values.astype(np.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {'x': self.features[idx], 'y': self.targets[idx]}

class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test,
                 rep_col_name, target_col_names,
                 batch_size=32, df_extra_test=None):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.rep_col_name = rep_col_name
        self.target_col_names = target_col_names
        self.batch_size = batch_size
        self.df_extra_test = df_extra_test
        self.extra_test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MultiTaskDataset(self.df_train,
                                              self.rep_col_name,
                                              self.target_col_names)
        self.val_dataset = MultiTaskDataset(self.df_val,
                                            self.rep_col_name,
                                            self.target_col_names)
        self.test_dataset = MultiTaskDataset(self.df_test,
                                             self.rep_col_name,
                                             self.target_col_names)
        if self.df_extra_test is not None and len(self.df_extra_test) > 0:
            self.extra_test_dataset = MultiTaskDataset(self.df_extra_test,
                                                       self.rep_col_name,
                                                       self.target_col_names)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def extra_test_dataloader(self):
        if self.extra_test_dataset is not None:
            return DataLoader(self.extra_test_dataset, batch_size=self.batch_size)
        return None

##############################################################################
# 4) Model + Architectures
##############################################################################

class MLPLayer(nn.Module):
    """
    One block of: Linear -> (BatchNorm) -> Activation -> (Dropout).
    """
    def __init__(self, in_dim, out_dim, activation, use_batch_norm=False, dropout_rate=0.0):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(in_dim, out_dim)
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

########################################################################
# Custom Losses
########################################################################

def exponential_loss(preds, y):
    # Simple placeholder example
    return torch.mean(torch.exp(preds) - y * preds)

def focal_loss(preds, y, alpha=1.0, gamma=2.0):
    p = torch.sigmoid(preds)
    ce = F.binary_cross_entropy_with_logits(preds, y, reduction='none')
    pt = y * p + (1 - y) * (1 - p)
    focal_factor = (1 - pt).pow(gamma)
    return alpha * torch.mean(focal_factor * ce)

def savage_loss(preds, y):
    bce = F.binary_cross_entropy_with_logits(preds, y, reduction='none')
    return torch.mean(bce ** 2)

def brier_score_loss(preds, y):
    p = torch.sigmoid(preds)
    return torch.mean((p - y) ** 2)

########################################################################
# CrossStitchUnit for multi-task cross-stitch
########################################################################

class CrossStitchUnit(nn.Module):
    """
    A cross-stitch unit for T tasks. We store a T x T parameter matrix,
    which linearly mixes the T input feature vectors.
    """

    def __init__(self, num_tasks):
        super().__init__()
        # Initialize near identity
        stitch = torch.eye(num_tasks) * 0.9 + 0.1 / num_tasks
        self.cross_stitch = nn.Parameter(stitch)

    def forward(self, *task_features):
        """
        task_features: list of length T,
          each is (batch_size, hidden_dim)
        Return the same length T, each (batch_size, hidden_dim),
          but linearly mixed.
        """
        feats = torch.stack(task_features, dim=1)  # (batch_size, T, hidden_dim)
        mixed = torch.einsum("tj,bjh->bth", self.cross_stitch, feats)
        out_list = []
        for t in range(mixed.shape[1]):
            out_list.append(mixed[:, t, :])
        return out_list

########################################################################
# SluiceUnit for multi-task sluice
########################################################################

class SluiceUnit(nn.Module):
    """
    A simplified Sluice mixing for T tasks. We keep a T x T alpha matrix
    that decides how to mix each pair of streams after each layer.
    """

    def __init__(self, num_tasks):
        super().__init__()
        # We'll do a T x T param matrix, with diagonal ~0.5 and off-diag ~0.5/(T-1).
        base = 0.5 * torch.eye(num_tasks)
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    base[i, j] = 0.5 / (num_tasks - 1)
        self.alpha = nn.Parameter(base)

    def forward(self, *task_features):
        feats = torch.stack(task_features, dim=1)  # (batch, T, hid)
        mixed = torch.einsum("tj,bjh->bth", self.alpha, feats)
        return [mixed[:, t, :] for t in range(mixed.shape[1])]

########################################################################
# Main MultiTaskFeedForward
########################################################################

class MultiTaskFeedForward(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dim_size: int = 128,
        dim_shrinking_scale: float = 0.5,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        L2_weight_norm: float = 0.0,
        L1_weight_norm: float = 0.0,
        activation_function: str = 'relu',
        model_type: str = 'shared_parameter',
        num_experts: int = 2,
        dropout_rate: float = 0.0,
        use_residual: bool = False,
        loss_type: str = 'bce_with_logits',
        use_batch_norm: bool = False,
        scheduler_step_size: int = 5,
        scheduler_gamma: float = 0.5,
        private_loss_scale: float = 1.0,
        task1_loss_scale: float = 1.0,
        task2_loss_scale: float = 1.0,
        task3_loss_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Basic parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_size = dim_size
        self.dim_shrinking_scale = dim_shrinking_scale
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.L2_weight_norm = L2_weight_norm
        self.L1_weight_norm = L1_weight_norm
        self.activation_function = activation_function
        self.model_type = model_type
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.loss_type = loss_type
        self.use_batch_norm = use_batch_norm

        # LR scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # Scale for private weights
        self.private_loss_scale = private_loss_scale

        self.task1_loss_scale = task1_loss_scale
        self.task2_loss_scale = task2_loss_scale
        self.task3_loss_scale = task3_loss_scale

        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.act_fn = activations.get(self.activation_function, nn.ReLU())

        # Build the appropriate network architecture
        self._build_model()
        
    def _build_shared_layers(self):
        """
        Returns a ModuleList of MLPLayer blocks.
        If you're using residual connections, these layers
        can be processed in _forward_shared_layers().
        """
        layers = nn.ModuleList()
        in_dim = self.input_dim
        current_dim = self.dim_size
    
        for _ in range(self.num_layers):
            block = MLPLayer(
                in_dim=in_dim,
                out_dim=current_dim,
                activation=self.act_fn,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate
            )
            layers.append(block)
            in_dim = current_dim
            current_dim = max(1, int(current_dim * self.dim_shrinking_scale))
    
        return layers

    def _build_model(self):
        if self.model_type == 'shared_parameter':
            self._build_shared_parameter()
        elif self.model_type == 'MMoE':
            self._build_mmoe()
        elif self.model_type == 'dirty_method':
            self._build_dirty_method()
        elif self.model_type == 'cross_stitch':
            self._build_cross_stitch_multi()
        elif self.model_type == 'sluice':
            self._build_sluice_multi()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    ########################################################################
    # Architecture Builders
    ########################################################################
    def _build_shared_parameter(self):
        self.shared_layers = self._build_shared_layers()
        final_dim = self._get_final_dim()
        self.head = nn.Linear(final_dim, self.output_dim)

    def _build_mmoe(self):
        self.experts = nn.ModuleList([self._build_shared_layers() for _ in range(self.num_experts)])
        final_dim = self._get_final_dim()
        self.towers = nn.ModuleList([nn.Linear(final_dim, 1) for _ in range(self.output_dim)])
        self.gates = nn.ModuleList([
            nn.Linear(self.input_dim, self.num_experts) for _ in range(self.output_dim)
        ])

    def _build_dirty_method(self):
        # shared trunk
        self.shared_layers = self._build_shared_layers()
        self.shared_final_dim = self._get_final_dim()
        self.private_layers = nn.ModuleList()
        self.task_heads = nn.ModuleList()

        # We have one private layer + head per task
        for _ in range(self.output_dim):
            private_layer = MLPLayer(
                in_dim=self.shared_final_dim,
                out_dim=self.shared_final_dim,
                activation=self.act_fn,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate
            )
            self.private_layers.append(private_layer)
            head_in_dim = 2 * self.shared_final_dim
            self.task_heads.append(nn.Linear(head_in_dim, 1))

    def _build_cross_stitch_multi(self):
        T = self.output_dim
        self.streams = nn.ModuleList([
            nn.ModuleList([
                MLPLayer(
                    in_dim=self.input_dim if layer_idx == 0 else self._get_layer_dim(layer_idx - 1),
                    out_dim=self._get_layer_dim(layer_idx),
                    activation=self.act_fn,
                    use_batch_norm=self.use_batch_norm,
                    dropout_rate=self.dropout_rate
                )
                for layer_idx in range(self.num_layers)
            ])
            for _ in range(T)
        ])
        self.cs_units = nn.ModuleList([CrossStitchUnit(T) for _ in range(self.num_layers)])
        final_dim = self._get_final_dim()
        self.heads = nn.ModuleList([nn.Linear(final_dim, 1) for _ in range(T)])

    def _build_sluice_multi(self):
        T = self.output_dim
        self.streams = nn.ModuleList([
            nn.ModuleList([
                MLPLayer(
                    in_dim=self.input_dim if layer_idx == 0 else self._get_layer_dim(layer_idx - 1),
                    out_dim=self._get_layer_dim(layer_idx),
                    activation=self.act_fn,
                    use_batch_norm=self.use_batch_norm,
                    dropout_rate=self.dropout_rate
                )
                for layer_idx in range(self.num_layers)
            ])
            for _ in range(T)
        ])
        self.sluice_units = nn.ModuleList([SluiceUnit(T) for _ in range(self.num_layers)])
        final_dim = self._get_final_dim()
        self.heads = nn.ModuleList([nn.Linear(final_dim, 1) for _ in range(T)])

    ########################################################################
    # Helpers
    ########################################################################
    def _get_layer_dim(self, layer_idx):
        dim = self.dim_size
        for _ in range(layer_idx):
            dim = max(1, int(dim * self.dim_shrinking_scale))
        return dim

    def _get_final_dim(self):
        dim = self.dim_size
        for i in range(1, self.num_layers):
            dim = max(1, int(dim * self.dim_shrinking_scale))
        return dim

    ########################################################################
    # Forward
    ########################################################################
    def forward(self, x):
        if self.model_type == 'shared_parameter':
            return self._forward_shared_parameter(x)
        elif self.model_type == 'MMoE':
            return self._forward_mmoe(x)
        elif self.model_type == 'dirty_method':
            return self._forward_dirty_method(x)
        elif self.model_type == 'cross_stitch':
            return self._forward_cross_stitch_multi(x)
        elif self.model_type == 'sluice':
            return self._forward_sluice_multi(x)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _forward_shared_parameter(self, x):
        out = self._forward_shared_layers(x, self.shared_layers)
        return self.head(out)

    def _forward_mmoe(self, x):
        expert_outputs = []
        for expert_module in self.experts:
            expert_out = self._forward_shared_layers(x, expert_module)
            expert_outputs.append(expert_out)
        final_outputs = []
        for i in range(self.output_dim):
            gate_scores = F.softmax(self.gates[i](x), dim=1)
            # Stack all expert outputs => shape: (B, num_experts, hidden_dim)
            stacked = torch.stack(expert_outputs, dim=1)
            # Weighted combination of experts
            weighted = torch.einsum('be,beh->bh', gate_scores, stacked)
            out = self.towers[i](weighted)
            final_outputs.append(out)
        return torch.cat(final_outputs, dim=1)

    def _forward_dirty_method(self, x):
        shared_out = self._forward_shared_layers(x, self.shared_layers)
        outputs = []
        for private_layer, head in zip(self.private_layers, self.task_heads):
            private_out = private_layer(shared_out)
            combined = torch.cat([shared_out, private_out], dim=1)
            out = head(combined)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def _forward_cross_stitch_multi(self, x):
        T = self.output_dim
        feats = [x.clone() for _ in range(T)]
        for layer_idx in range(self.num_layers):
            new_feats = []
            for t in range(T):
                out_t = self.streams[t][layer_idx](feats[t])
                new_feats.append(out_t)
            new_feats = self.cs_units[layer_idx](*new_feats)
            feats = new_feats
        outputs = []
        for t in range(T):
            outputs.append(self.heads[t](feats[t]))
        return torch.cat(outputs, dim=1)

    def _forward_sluice_multi(self, x):
        T = self.output_dim
        feats = [x.clone() for _ in range(T)]
        for layer_idx in range(self.num_layers):
            new_feats = []
            for t in range(T):
                out_t = self.streams[t][layer_idx](feats[t])
                new_feats.append(out_t)
            new_feats = self.sluice_units[layer_idx](*new_feats)
            feats = new_feats
        outputs = []
        for t in range(T):
            outputs.append(self.heads[t](feats[t]))
        return torch.cat(outputs, dim=1)

    def _forward_shared_layers(self, x, layers):
        out = x
        for layer in layers:
            new_out = layer(out)
            if self.use_residual and (new_out.shape == out.shape):
                out = out + new_out
            else:
                out = new_out
        return out

    ########################################################################
    # Compute Loss with Partial Labels
    ########################################################################
    def _compute_loss(self, preds, y):
        """
        Handles partially unknown targets: for any y[i, t] == -1, 
        skip that sample for task t.
        Also incorporates task1_loss_scale, task2_loss_scale, and task3_loss_scale.
        """
        device = preds.device
        total_loss = torch.tensor(0.0, device=device)
        scale_sum = 0.0

        for t in range(self.output_dim):
            # Mask out unknown labels
            mask = (y[:, t] >= 0.0)
            if mask.any():
                preds_t = preds[mask, t]
                y_t = y[mask, t]

                if self.loss_type == "bce_with_logits":
                    loss_t = F.binary_cross_entropy_with_logits(preds_t, y_t, reduction='mean')
                elif self.loss_type == "exponential_loss":
                    loss_t = exponential_loss(preds_t, y_t)
                elif self.loss_type == "focal_loss":
                    loss_t = focal_loss(preds_t, y_t)
                elif self.loss_type == "savage_loss":
                    loss_t = savage_loss(preds_t, y_t)
                elif self.loss_type == "brier_score":
                    loss_t = brier_score_loss(preds_t, y_t)
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")

                # ---------------- Scale each task's loss accordingly ----------------
                if t == 0:
                    scale = self.task1_loss_scale
                elif t == 1:
                    scale = self.task2_loss_scale
                elif t == 2:
                    scale = self.task3_loss_scale
                else:
                    # For extra tasks beyond T=3, default to 1.0 or define your own
                    scale = 1.0

                total_loss = total_loss + scale * loss_t
                scale_sum += scale

        if scale_sum > 0:
            total_loss = total_loss / scale_sum
        else:
            # If there are no valid labels in this batch, produce 0 but keep gradient
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss

    ###########################
    # Lightning steps
    ###########################
    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        preds = self(x)
        loss = self._compute_loss(preds, y)

        # -------------------------------------------------------------------
        # L1/L2 penalty modifications if using "dirty_method"
        # -------------------------------------------------------------------
        if self.model_type == "dirty_method":
            shared_params = []
            private_params = []
            # We'll identify "private" by looking for 'private_layers' or 'task_heads'
            # in the parameter names. Everything else is "shared".
            for name, param in self.named_parameters():
                if any(s in name for s in ["private_layers", "task_heads"]):
                    private_params.append(param)
                else:
                    shared_params.append(param)

            # -- L1 penalty:
            l1_shared = sum(p.abs().sum() for p in shared_params)
            l1_private = sum(p.abs().sum() for p in private_params)

            # -- L2 penalty:
            l2_shared = sum((p**2).sum() for p in shared_params)
            l2_private = sum((p**2).sum() for p in private_params)

            # Combine them, scaling the private part
            total_l1_penalty = self.L1_weight_norm * l1_shared + \
                               (self.L1_weight_norm * self.private_loss_scale) * l1_private
            total_l2_penalty = self.L2_weight_norm * l2_shared + \
                               (self.L2_weight_norm * self.private_loss_scale) * l2_private

            # Add to final loss
            loss = loss + total_l1_penalty + total_l2_penalty
        else:
            # Simpler single L1 penalty for everything
            if self.L1_weight_norm > 0:
                l1_penalty = sum(param.abs().sum() for param in self.parameters())
                loss = loss + self.L1_weight_norm * l1_penalty
            # If you wish to do L2 here, you can similarly add it.

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        preds = self(x)
        val_loss = self._compute_loss(preds, y)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        test_loss = self._compute_loss(self(x), y)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        # If you're manually calculating L2, set weight_decay=0 here
        # so you don't double-penalize. Otherwise you can combine the effects.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0  # <--- optional if using manual L2 above
        )
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma
            ),
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler_config]

##############################################################################
# 5) MultiTaskEvaluationCallback
##############################################################################

def compute_auroc_auprc(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    is_logits = (model.hparams.loss_type in ["bce_with_logits",
                                             "exponential_loss",
                                             "focal_loss",
                                             "savage_loss",
                                             "brier_score"])

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["x"], batch["y"]
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            if is_logits:
                preds = torch.sigmoid(preds)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    num_tasks = all_targets.shape[1]
    aurocs = []
    auprcs = []
    for task_idx in range(num_tasks):
        valid_mask = (all_targets[:, task_idx] >= 0.0)
        y_true = all_targets[valid_mask, task_idx]
        y_pred = all_preds[valid_mask, task_idx]
        if len(y_true) == 0:
            auroc_val, auprc_val = float('nan'), float('nan')
        else:
            try:
                auroc_val = roc_auc_score(y_true, y_pred)
                auprc_val = average_precision_score(y_true, y_pred)
            except ValueError:
                auroc_val = float('nan')
                auprc_val = float('nan')
        aurocs.append(auroc_val)
        auprcs.append(auprc_val)

    print(f"auROCs are {aurocs}")
    print(f"auPRCs are {auprcs}")

    return {
        "auroc_per_task": [float(a) for a in aurocs],
        "auprc_per_task": [float(p) for p in auprcs],
    }

class MultiTaskEvaluationCallback(Callback):
    def __init__(self, save_filename="final_metrics.json"):
        super().__init__()
        self.save_filename = save_filename
        self.results_per_epoch = []

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        epoch_metrics = {"epoch": current_epoch}
        device = pl_module.device

        def store_per_output_tensors(prefix, values):
            if values is None:
                return
            for i, val in enumerate(values):
                name = f"{prefix}_output_{i}"
                trainer.callback_metrics[name] = torch.tensor(val, device=device, dtype=torch.float)

        train_loader = trainer.datamodule.train_dataloader()
        train_stats = compute_auroc_auprc(pl_module, train_loader, device)
        epoch_metrics["train_auroc"] = train_stats["auroc_per_task"]
        epoch_metrics["train_auprc"] = train_stats["auprc_per_task"]

        val_loader = trainer.datamodule.val_dataloader()
        val_stats = compute_auroc_auprc(pl_module, val_loader, device)
        epoch_metrics["val_auroc"] = val_stats["auroc_per_task"]
        epoch_metrics["val_auprc"] = val_stats["auprc_per_task"]

        test_loader = trainer.datamodule.test_dataloader()
        test_stats = compute_auroc_auprc(pl_module, test_loader, device)
        epoch_metrics["test_auroc"] = test_stats["auroc_per_task"]
        epoch_metrics["test_auprc"] = test_stats["auprc_per_task"]

        extra_test_loader = trainer.datamodule.extra_test_dataloader()
        if extra_test_loader:
            extra_stats = compute_auroc_auprc(pl_module, extra_test_loader, device)
            epoch_metrics["extra_test_auroc"] = extra_stats["auroc_per_task"]
            epoch_metrics["extra_test_auprc"] = extra_stats["auprc_per_task"]
        else:
            epoch_metrics["extra_test_auroc"] = None
            epoch_metrics["extra_test_auprc"] = None

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            epoch_metrics["val_loss"] = float(val_loss.cpu().numpy())

        val_auprc_values = epoch_metrics["val_auprc"]
        if val_auprc_values is not None:
            val_auprc_avg = float(np.nanmean(val_auprc_values))
            epoch_metrics["val_auprc_avg"] = val_auprc_avg
            trainer.callback_metrics["val_auprc_avg"] = torch.tensor(val_auprc_avg, device=device, dtype=torch.float)

        print(f"[Epoch {current_epoch}] metrics: {epoch_metrics}")
        self.results_per_epoch.append(epoch_metrics)

        store_per_output_tensors("train_auroc", epoch_metrics["train_auroc"])
        store_per_output_tensors("train_auprc", epoch_metrics["train_auprc"])
        store_per_output_tensors("val_auroc", epoch_metrics["val_auroc"])
        store_per_output_tensors("val_auprc", epoch_metrics["val_auprc"])
        store_per_output_tensors("test_auroc", epoch_metrics["test_auroc"])
        store_per_output_tensors("test_auprc", epoch_metrics["test_auprc"])
        store_per_output_tensors("extra_test_auroc", epoch_metrics["extra_test_auroc"])
        store_per_output_tensors("extra_test_auprc", epoch_metrics["extra_test_auprc"])

    def on_fit_end(self, trainer, pl_module):
        if not self.results_per_epoch:
            print("No epochs recorded; skipping final metrics reporting.")
            return

        final_epoch_metrics = self.results_per_epoch[-1]
        hparams = dict(pl_module.hparams)
        final_dict = {
            "hyperparameters": hparams,
            "epoch_metrics": self.results_per_epoch
        }

        save_path = os.path.join(os.getcwd(), self.save_filename)
        with open(save_path, "w") as f:
            json.dump(final_dict, f, indent=4)
        print(f"\nSaved final results (hyperparams + metrics) to {save_path}\n")

        final_report_dict = {}

        def store_per_output_columns(prefix, values):
            if values is None:
                return
            for i, val in enumerate(values):
                name = f"{prefix}_output_{i}"
                final_report_dict[name] = val

        store_per_output_columns("train_auroc", final_epoch_metrics["train_auroc"])
        store_per_output_columns("train_auprc", final_epoch_metrics["train_auprc"])
        store_per_output_columns("val_auroc", final_epoch_metrics["val_auroc"])
        store_per_output_columns("val_auprc", final_epoch_metrics["val_auprc"])
        store_per_output_columns("test_auroc", final_epoch_metrics["test_auroc"])
        store_per_output_columns("test_auprc", final_epoch_metrics["test_auprc"])
        store_per_output_columns("extra_test_auroc", final_epoch_metrics["extra_test_auroc"])
        store_per_output_columns("extra_test_auprc", final_epoch_metrics["extra_test_auprc"])

        final_report_dict["val_loss"] = final_epoch_metrics.get("val_loss", None)
        final_report_dict["val_auprc_avg"] = final_epoch_metrics.get("val_auprc_avg", None)

        from ray import tune
        tune.report(metrics=final_report_dict)

##############################################################################
# 6) train_with_tune + main()
##############################################################################

def train_with_tune(config,
                    df_train,
                    df_val,
                    df_test,
                    rep_col,
                    target_cols,
                    max_epochs=10,
                    df_extra_test=None):

    data_module = MultiTaskDataModule(
        df_train, df_val, df_test,
        rep_col_name=rep_col,
        target_col_names=target_cols,
        batch_size=config["batch_size"],
        df_extra_test=df_extra_test
    )
    data_module.setup()
    input_dim = data_module.train_dataset.features.shape[1]

    model = MultiTaskFeedForward(
        input_dim=input_dim,
        output_dim=len(target_cols),
        dim_size=config["dim_size"],
        dim_shrinking_scale=config["dim_shrinking_scale"],
        num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        L2_weight_norm=config["L2_weight_norm"],
        L1_weight_norm=config["L1_weight_norm"],
        activation_function=config["activation_function"],
        model_type=config["model_type"],
        num_experts=2,
        dropout_rate=config["dropout_rate"],
        use_residual=config["use_residual"],
        loss_type=config["loss_type"],
        use_batch_norm=config["use_batch_norm"],
        scheduler_step_size=config["scheduler_step_size"],
        scheduler_gamma=config["scheduler_gamma"],
        private_loss_scale=config.get("private_loss_scale", 1.0),
        task1_loss_scale=config.get("task1_loss_scale", 1.0),
        task2_loss_scale=config.get("task2_loss_scale", 1.0),
        task3_loss_scale=config.get("task3_loss_scale", 1.0),
    )

    tune_callback = TuneReportCallback(
        metrics={
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "val_auprc_avg": "val_auprc_avg",
            "val_auprc_output_0": "val_auprc_output_0",
            "val_auprc_output_1": "val_auprc_output_1", 
            "val_auprc_output_2": "val_auprc_output_2"
        },
        on="validation_epoch_end",
    )

    eval_callback = MultiTaskEvaluationCallback(save_filename="final_metrics.json")

    es_metric = config.get("early_stop_metric", "val_auprc_avg")
    mode = "min" if es_metric == "val_loss" else "max"

    early_stop_callback = EarlyStopping(
        monitor=es_metric,
        patience=config["patience"],
        mode=mode,
        verbose=True
    )

    model_ckpt_callback = ModelCheckpoint(
        dirpath=config.get("model_file_path", "./checkpoints"),
        monitor=es_metric,
        mode=mode,
        save_top_k=1,
        filename="best-{epoch}-{" + es_metric + ":.3f}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            eval_callback,
            tune_callback,
            model_ckpt_callback,
            early_stop_callback
        ],
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=config.get("model_file_path", None),
    )

    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module, ckpt_path=None)

class AppendMetricsCallback(tune.Callback):
    """
    A custom Ray Tune callback that appends each trial's final result
    to an ongoing summary_metrics.tsv, so you can watch it build up over time.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._header_written = False

    def on_trial_complete(self, iteration, trials, trial, **info):
        """
        Called when a trial finishes. We'll append one row with that trial's final results.
        """
        last_result = trial.last_result.copy()
        
        row_df = pd.DataFrame([last_result])
        
        with open(self.csv_path, "a") as f:
            row_df.to_csv(f, sep="\t", index=False, header=not self._header_written)
        self._header_written = True

def main():
    
    #==========================================================================
    # This parameter affects the total number of hyperparamter combinations
    # raytune will consider before stopping. If you want to just decide when
    # the training should stop manually by watching when you think the metrics
    # have converged, you can also just set this to a very large number. Model
    # saving is done after every round of training.
    #==========================================================================
    
    # Number of hyperparameters combinations tried during training.
    NUM_TRIALS = 1000
    
    #==========================================================================
    # This parameter refers to the file path for the dataset used to generate
    # train, val, and test splits. You also must define the column name for the
    # column with the molecular representation (script has so far been run with
    # pythonic lists for the minimol representations in the "Minimol_Fingerprint"
    # column, however other formats may also be possible.
    #==========================================================================
    
    # Path to the dataset used for model fitting.
    file_path = (
        "/path/to/my/fav/dataset.tsv"
    )
    
    #--------------------------------------------------------------------------
    
    # Column name for molecular representations.
    rep_col_name = "Minimol_Fingerprint"
    
    #--------------------------------------------------------------------------
    
    # Column names for the different targets.
    target_col_names = [
        "HepG2_10uM_Binarized",
        "HSkMC_10uM_Binarized",
        "IMR90_10uM_Binarized",
    ]
    
    #--------------------------------------------------------------------------
    
    # Data loading and train, val, test split generation. The second argument
    # states the fraction of the dataste which is split into training,
    # validation, and test datasets.
    df = read_data(file_path, rep_col_name, target_col_names)
    df_train, df_val, df_test = train_val_test_split(df, (0.80, 0.10, 0.10))
    
    #--------------------------------------------------------------------------
    
    # This is the path to an extra dataset if you want to watch how well the
    # the model does on it.
    extra_test_set_file_path = (
        "/Users/asselism/Desktop/Collins_Lab/Datasets/Organized_by_Data_Type/ADMET_Data/Organized_by_Data_Type/Toxicity/Organized_by_Source/ECBD/ECBD_HepG2_100K/Original_Datasets/ECBD_HepG2_STL_tanimoto_extra_test_dataset.tsv"
    )
    
    #--------------------------------------------------------------------------
    
    # Analogous variables for interpretting the extra test dataset.
    extra_test_set_rep_col_name = "Minimol_Fingerprint"
    extra_test_set_target_col_names = [
        "HepG2_10uM_Binarized",
        "HSkMC_10uM_Binarized",
        "IMR90_10uM_Binarized",
    ]

    if extra_test_set_file_path:
        df_extra_test = read_data(
            extra_test_set_file_path,
            extra_test_set_rep_col_name,
            extra_test_set_target_col_names
        )
    else:
        df_extra_test = None

    if not ray.is_initialized():
        ray.init()
        
    #==========================================================================
    # The following is the hyperparameter space search during model
    # hyperparameter tuning. There are some less commonly used arguments here
    # for completeness, which have been set to a single value. An option is to
    # leave some of there hyperparameters only a sinlge "choice" which would
    # essentially exclude them from the search space.
    #==========================================================================

    search_space = {
        # This is just the width of the hidden layers of a model.
        "dim_size": tune.lograndint(128, 4096),
        # This handles the situation where you would like for subsequent layers
        # to be smaller than previous layers in the model architecture.
        "dim_shrinking_scale": tune.uniform(0.5, 1.0),
        # This is just the number of hidden layers.
        "num_layers": tune.randint(1, 3),
        # This is the step size for the optimization algorithm.
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        # This is the batch size used during model fitting.
        "batch_size": tune.choice([32, 64, 128]),
        # This is the L2 norm for the weights.
        "L2_weight_norm": tune.loguniform(1e-8, 1e-2),
        # This L1 weight norm places a sparsity prior on the weights.
        "L1_weight_norm": tune.loguniform(1e-12, 1e-2),
        # This lets you change the activation function. Generally, relu is fine.
        "activation_function": tune.choice(["relu", "tanh", "leaky_relu"]),
        # This lets you explores some different MTL architectures.
        "model_type": tune.choice(["shared_parameter", "dirty_method"]),
        # This is how many rounds does the validation need to not improve for
        # early termination of model training.
        "patience": tune.choice([2]),
        # This is the parameter for dropout normalization. Set to 0 to not use
        # dropout.
        "dropout_rate": tune.uniform(0.0, 0.9),
        # This controls whether or not a residual is passed through the network.
        "use_residual": tune.choice([True, False]),
        # This controls whether or not batch normalization is used.
        "use_batch_norm": tune.choice([True, False]),
        # This controls the loss function used during model training.
        "loss_type": tune.choice(["bce_with_logits"]),
        # This controls what metric is used for early stopping.
        "early_stop_metric": tune.choice(["val_loss"]),
        # This controls the number of epochs that must pass before the gradient
        # descent step size is decreased by a factor.
        "scheduler_step_size": tune.lograndint(2, 20),
        # This is the magnitude of that factor.
        "scheduler_gamma": tune.uniform(0.1, 0.9),
        # This is the ratio of normalization imposed on task-specific versus
        # task-shared weights.
        "private_loss_scale": tune.loguniform(1e-1, 10.0),
        # These are factors placed on the classification loss for each task.
        "task1_loss_scale": tune.loguniform(0.7, 1.4),
        "task2_loss_scale": tune.loguniform(0.7, 1.4),
        "task3_loss_scale": tune.loguniform(0.7, 1.4),
        # This is just the path to the file where trained models are saved.
        # This honestly shouldn't be a hyperparameter and will change.
        "model_file_path": tune.choice(["/path/to/where/I/want/my/models/to/be/"])
    }

    algo = OptunaSearch(metric="val_auprc_avg", mode="max")

    summary_metrics_path = "/path/to/where/I/want/my/metrics/to/be/saved/to/summary_metrics.tsv"
    append_metrics_callback = AppendMetricsCallback(csv_path=summary_metrics_path)

    tuner = tune.run(
        tune.with_parameters(
            train_with_tune,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            rep_col=rep_col_name,
            target_cols=target_col_names,
            max_epochs=25, # This is the number of epochs each model is allowed
            # to train on before training is terminated (normally, early
            # stopping ensures this number is not reached).
            df_extra_test=df_extra_test
        ),
        config=search_space,
        search_alg=algo,
        num_samples=NUM_TRIALS,
        resources_per_trial={"cpu": 1, "gpu": 0},
        metric="val_auprc_avg",  # We monitor this metric for "best" in Ray
        mode="max",
        callbacks=[append_metrics_callback]
    )

    best_config = tuner.get_best_config(metric="val_auprc_avg", mode="max")
    print("Best hyperparameters found were:", best_config)

if __name__ == "__main__":
    main()