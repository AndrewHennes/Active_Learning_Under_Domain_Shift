#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference pipeline – streamed, memory-safe, multi-CPU
Created on Sat Feb 15 12:22:59 2025
@author: asselism

Activate environment:
    source /Users/asselism/Desktop/Collins_Lab/Environments/Environments_for_Models/Environment_for_Torch/FNN/pytorch_venv/bin/activate
"""

from __future__ import annotations

import ast
import json
import os
import warnings
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION  –  ***  EDIT THESE THREE PATHS ONLY  ***
# ──────────────────────────────────────────────────────────────────────
RAW_TSV = (
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Minimol_Fingerprints.tsv"
)  # ← original input

ACTIVITY_TSV = (
    "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Minimol_Fingerprints_Intermediate_Timed.tsv"
)  # ← intermediate

FINAL_TSV = (
   "/Users/asselism/Desktop/Collins_Lab/Analysis/Organized_by_Project/Non-toxic_Antibiotics/Analyses_for_Paper/ECBD_Cytotoxicity/Formatted_ECBD_Antibiotic_Toxicity_Data_Combined_with_Minimol_Fingerprints_with_Tox_and_Antibiotic_Scores_Timed.tsv"
)  # ← final output

# ──────────────────────────────────────────────────────────────────────
#  PARALLELISM (override via env-vars if you want)
# ──────────────────────────────────────────────────────────────────────
N_WORKERS: int = int(os.getenv("N_WORKERS", "0")) or (os.cpu_count() or 1)
USE_PROCESSES: bool = bool(int(os.getenv("USE_PROCESSES", "0")))  # 0 = threads (default)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "50000"))  # rows per chunk
REPORT_EVERY: int = 100_000                               # status print cadence

# ──────────────────────────────────────────────────────────────────────
# DIRECTORIES HOLDING CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────
tox_models_dir = (
    "/Users/asselism/Desktop/Collins_Lab/Repositories/"
    "Nontoxic_Antibiotics_Project/Model_Checkpoints/Tox_Checkpoints/MTL_Models/"
)
antibiotic_model_dir = (
    "/Users/asselism/Desktop/Collins_Lab/Repositories/Active_Learning/ActiveLearning/Models/Round_1_Models/Antibiotic_Activity_Model_Checkpoints/"
)

# ──────────────────────────────────────────────────────────────────────
#  Progress-bar helper (uses tqdm if available, else silent fallback)
# ──────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:  # fallback to no-op if tqdm is not installed

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

# ──────────────────────────────────────────────────────────────────────
#  COLUMN NAME FOR MINIMOL VECTORS
# ──────────────────────────────────────────────────────────────────────
MINIMOL_COL = "Minimol_Representation"   # change here if your TSV uses a different header

# =====================================================================
# 1)  ANTIBIOTIC-ACTIVITY PREDICTIONS  (original definition, unchanged)
# =====================================================================
def FNN_with_attention(
    # Data / IO
    train_representations: Optional[np.ndarray] = None,
    train_targets: Optional[np.ndarray] = None,
    test_representations: Optional[np.ndarray] = None,
    test_targets: Optional[np.ndarray] = None,
    score_data: Optional[List[dict]] = None,
    f_in_path: Optional[str] = None,
    f_out: Optional[str] = None,
    # Hyper-params
    loss_function: Optional[str] = None,
    batch_size: int = 300,
    hidden_dim: int = 1024,
    learning_rate: float = 0.1,
    n_iters: int = 1000,
    model_path: Optional[str] = None,
    num_layers: int = 2,
    dropout_prob: float = 0.1,
    mode: str = "train",  # {"train", "evaluate", "score"}
    patience: int = 10,
    lr_decay_step: int = 10,
    lr_decay_gamma: float = 0.1,
    model_type: str = "attention",  # {"attention", "mlp"}
    weight_decay: float = 1e-4,
    num_heads: int = 4,
):
    """
    Flexible feed-forward network with optional single-token self-attention.
    Only ``mode="score"`` is used by this pipeline.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ─────────────────────────────────────────────
    #  Internal model definitions
    # ─────────────────────────────────────────────
    class TaskHeadWithAttention(nn.Module):
        """MLP + skip connection + multi-head self-attention (single token)."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
            num_heads: int = 4,
        ):
            super().__init__()
            assert input_dim % num_heads == 0, "`input_dim` must be divisible by `num_heads`"

            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, batch_first=True
            )

            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.final_dense = nn.Linear(input_dim + hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            attn_out, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            x = attn_out.squeeze(1)

            for dense, bn in zip(self.layers, self.batch_norms):
                x = self.dropout(F.relu(bn(dense(x))))

            return self.final_dense(torch.cat([residual, x], dim=1))

    class TaskHead(nn.Module):
        """Plain MLP with skip connection."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
        ):
            super().__init__()
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            self.final_dense = nn.Linear(input_dim + hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            for dense, bn in zip(self.layers, self.batch_norms):
                x = self.dropout(F.relu(bn(dense(x))))
            return self.final_dense(torch.cat([residual, x], dim=1))

    # ─────────────────────────────────────────────
    #  MODE: SCORE
    # ─────────────────────────────────────────────
    if mode == "score":
        if f_in_path is None or not os.path.exists(f_in_path):
            raise ValueError("'score' mode requires a valid `f_in_path`")
        if score_data is None:
            raise ValueError("'score' mode requires `score_data`")

        # ---------- load checkpoint & infer architecture -----------
        state_dict = torch.load(
            f_in_path, map_location="cpu", weights_only=True  # suppress pickle warning
        )
        layer0 = next(k for k in state_dict if k.endswith("layers.0.weight"))
        input_dim = state_dict[layer0].shape[1]
        hidden_dim_ckpt = state_dict[layer0].shape[0]
        output_dim = state_dict["final_dense.weight"].shape[0]
        num_layers_ckpt = len(
            {k.split(".")[1] for k in state_dict if k.startswith("layers.") and ".weight" in k}
        )

        if any("attention.in_proj_weight" in k for k in state_dict):
            model = TaskHeadWithAttention(
                input_dim,
                output_dim,
                hidden_dim_ckpt,
                num_layers=num_layers_ckpt,
                dropout_rate=dropout_prob,
                num_heads=num_heads,
            )
        else:
            model = TaskHead(
                input_dim,
                output_dim,
                hidden_dim_ckpt,
                num_layers=num_layers_ckpt,
                dropout_rate=dropout_prob,
            )

        model.load_state_dict(state_dict)
        model.eval()

        # ---------- prepare input ----------
        reps = [
            torch.as_tensor(d["featurized_smiles"], dtype=torch.float32)
            if not isinstance(d["featurized_smiles"], torch.Tensor)
            else d["featurized_smiles"].float()
            for d in score_data
        ]
        X = torch.stack(reps)
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

        preds: List[List[float]] = []
        for (batch_x,) in loader:
            with torch.no_grad():
                preds.extend(torch.sigmoid(model(batch_x)).cpu().tolist())

        # ---------- optional disk output ----------
        if f_out:
            with open(f_out, "w") as fh:
                fh.write("\t".join([f"pred_{i}" for i in range(output_dim)]) + "\n")
                for row in preds:
                    fh.write("\t".join(map(str, row)) + "\n")
            print(f"[activity] saved predictions → {f_out}")

        return preds

    raise NotImplementedError(f"Mode '{mode}' not implemented in this script.")

# ---------------------------------------------------------------------
#  Helper – safe vector parser  (replaces ``eval``)
# ---------------------------------------------------------------------
def safe_literal_vector(txt) -> Optional[List[float]]:
    """
    Return list[float] on success, else *None* (NaNs, bad JSON etc.).
    """
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return None
    if not isinstance(txt, str):
        return None
    try:
        return ast.literal_eval(txt)
    except Exception:
        try:
            return json.loads(txt)
        except Exception:
            return None

# ---------------------------------------------------------------------
#  ACTIVITY STAGE – streaming & multi-CPU
# ---------------------------------------------------------------------
def _score_model(model_file: str, score_data: list[dict], bs: int):
    preds = FNN_with_attention(
        mode="score", f_in_path=model_file, score_data=score_data, batch_size=bs
    )
    return model_file, preds


def stream_activity_on_tsv(
    raw_tsv: str,
    out_tsv: str,
    model_dir: str,
    chunk_size: int = CHUNK_SIZE,
    max_workers: int = N_WORKERS,
):
    model_files = sorted(glob(os.path.join(model_dir, "*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")

    Exec = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor

    rows_seen = rows_bad = 0
    header_written = False
    last_report = 0

    reader = pd.read_csv(raw_tsv, sep="\t", dtype=str, chunksize=chunk_size)

    for df in tqdm(reader, desc="Activity | TSV chunks"):
        vecs = df[MINIMOL_COL].map(safe_literal_vector)
        good = vecs.notnull()
        bad_here = (~good).sum()
        rows_seen += len(df)
        rows_bad += bad_here
        if bad_here:
            df = df.loc[good].copy()
            vecs = vecs.loc[good]
        if df.empty:
            continue

        score_data = [{"featurized_smiles": v} for v in vecs]

        preds_by_model: dict[str, list[list[float]]] = {}
        with Exec(max_workers=min(max_workers, len(model_files))) as pool:
            futs = {
                pool.submit(
                    _score_model,
                    mf,
                    score_data,
                    min(8192, len(score_data)),
                ): mf
                for mf in model_files
            }
            for fut in as_completed(futs):
                mf, preds = fut.result()
                preds_by_model[mf] = preds

        for mf, preds in preds_by_model.items():
            base = Path(mf).stem
            if len(preds[0]) == 1:
                df[f"{base}_output"] = [p[0] for p in preds]
            else:
                for i in range(len(preds[0])):
                    df[f"{base}_output_{i}"] = [p[i] for p in preds]

        df.to_csv(
            out_tsv,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

        if rows_seen - last_report >= REPORT_EVERY:
            tqdm.write(
                f"[activity] {rows_seen:,} rows  |  dropped {rows_bad:,} "
                f"({rows_bad / rows_seen:.2%})"
            )
            last_report = rows_seen

# ---------------------------------------------------------------------
#  TOXICITY STAGE – streamed, memory-safe
# ---------------------------------------------------------------------
def stream_tox_on_tsv(
    input_tsv: str,
    output_tsv: str,
    ckpt_dir: str,
    rep_col: str = MINIMOL_COL,
    chunk_size: int = CHUNK_SIZE,
    device: str = "cpu",
):
    ckpts = sorted(Path(ckpt_dir).glob("*.ckpt"))
    if not ckpts:
        warnings.warn("[tox] no checkpoints found – skipping")
        Path(output_tsv).write_text(Path(input_tsv).read_text())
        return

    tmp_out = Path(output_tsv).with_suffix(".tox_tmp.tsv")
    rows_seen = rows_bad = 0
    header_written = False
    last_report = 0

    reader = pd.read_csv(input_tsv, sep="\t", dtype=str, chunksize=chunk_size)

    for df in tqdm(reader, desc="Tox | TSV chunks"):
        vecs = df[rep_col].map(safe_literal_vector)
        good = vecs.notnull()
        bad_here = (~good).sum()
        rows_seen += len(df)
        rows_bad += bad_here
        if bad_here:
            df = df.loc[good].copy()
            vecs = vecs.loc[good]
        if df.empty:
            continue

        reps = torch.tensor(np.stack(vecs), dtype=torch.float32).to(device)

        for idx, ckpt in enumerate(ckpts):
            from Train_MTL_Tox_final_v1 import MultiTaskFeedForward  # local import
            model = (
                MultiTaskFeedForward.load_from_checkpoint(str(ckpt))
                .to(device)
                .eval()
            )
            preds_parts = []
            bs = 1024
            with torch.no_grad():
                for i in range(0, len(reps), bs):
                    preds_parts.append(torch.sigmoid(model(reps[i : i + bs])).cpu())
            preds = torch.cat(preds_parts).numpy()
            for t in range(preds.shape[1]):
                df[f"Ensemble_Member_{idx}_prediction_task_{t}"] = preds[:, t]
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        df.to_csv(
            tmp_out,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

        if rows_seen - last_report >= REPORT_EVERY:
            tqdm.write(
                f"[tox] {rows_seen:,} rows  |  dropped {rows_bad:,} "
                f"({rows_bad / rows_seen:.2%})"
            )
            last_report = rows_seen

    Path(output_tsv).write_text(Path(tmp_out).read_text())
    tmp_out.unlink(missing_ok=True)
    tqdm.write(
        f"[tox] done – total rows: {rows_seen:,}   dropped: {rows_bad:,} "
        f"({rows_bad / rows_seen:.2%})"
    )

# ---------------------------------------------------------------------
#  STREAMED SCORE AGGREGATION  – memory-safe
# ---------------------------------------------------------------------
def add_activity_and_tox_scores_streamed(
    tsv_in: Union[str, Path],
    tsv_out: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    na_strategy: str = "skip",
):
    """
    Read *tsv_in* in chunks, append summary columns, write to *tsv_out*.
    Uses O(chunk_size) memory.
    """
    species_prefix = {
        "Acinetobacter": "AB_AM_MTL_rep_{i}_col_1_is_pred_output_0",
        "Escherichia":   "EC_AM_MTL_rep_{i}_col_2_is_pred_output_1",
        "Klebsiella":    "KP_AM_MTL_rep_{i}_col_3_is_pred_output_2",
        "Pseudomonal":   "PA_AM_MTL_rep_{i}_col_4_is_pred_output_3",
    }
    tox_tasks = {0: "HepG2", 1: "HSkMC", 2: "IMR90"}

    header_written = False
    reader = pd.read_csv(tsv_in, sep="\t", chunksize=chunk_size, dtype=str)

    for chunk in tqdm(reader, desc="Aggregate | TSV chunks"):
        # convert only once per chunk – faster & less RAM
        chunk = chunk.apply(pd.to_numeric, errors="ignore")

        # 1) activity means
        for sp, pat in species_prefix.items():
            cols = [pat.format(i=r) for r in (1, 2, 3) if pat.format(i=r) in chunk.columns]
            if cols:
                chunk[f"Mean_{sp}_Activity"] = chunk[cols].astype(float).mean(
                    axis=1, skipna=True
                )
            elif na_strategy == "raise":
                raise KeyError(f"Missing columns for {sp}: {pat.format(i=1)} …")

        # 2) toxicity means
        for task_id, cell in tox_tasks.items():
            cols = [
                f"Ensemble_Member_{i}_prediction_task_{task_id}" for i in range(6)
                if f"Ensemble_Member_{i}_prediction_task_{task_id}" in chunk.columns
            ]
            if cols:
                chunk[f"Mean_{cell}_Toxicity"] = chunk[cols].astype(float).mean(
                    axis=1, skipna=True
                )
            elif na_strategy == "raise":
                raise KeyError(f"Missing Tox columns task {task_id}")

        # 3) per-row max tox & non-tox
        tox_mean_cols = [c for c in chunk.columns if c.startswith("Mean_") and c.endswith("_Toxicity")]
        if tox_mean_cols:
            chunk["Max_Tox_Score"] = chunk[tox_mean_cols].max(axis=1, skipna=True)
            chunk["Non-toxic Score"] = 1 - chunk["Max_Tox_Score"]

        # 4) flush
        chunk.to_csv(
            tsv_out,
            sep="\t",
            index=False,
            mode="w" if not header_written else "a",
            header=not header_written,
        )
        header_written = True

    tqdm.write(f"[aggregate] complete → {tsv_out}")


# =====================================================================
#  MAIN PIPELINE
# =====================================================================
if __name__ == "__main__":
    
    import time
    start_time = time.perf_counter()
    
    #1) activity predictions → ACTIVITY_TSV
    stream_activity_on_tsv(
        raw_tsv=RAW_TSV,
        out_tsv=ACTIVITY_TSV,
        model_dir=antibiotic_model_dir,
        chunk_size=CHUNK_SIZE,
        max_workers=N_WORKERS,
    )

    #2) toxicity ensemble (streamed) → same TSV
    stream_tox_on_tsv(
        input_tsv=ACTIVITY_TSV,
        output_tsv=ACTIVITY_TSV,   # overwrite in-place
        ckpt_dir=tox_models_dir,
        rep_col=MINIMOL_COL,
        chunk_size=CHUNK_SIZE,
        device="cpu",
    )

    # 3) aggregate scores → FINAL_TSV
    add_activity_and_tox_scores_streamed(
        tsv_in=ACTIVITY_TSV,
        tsv_out=FINAL_TSV,
        chunk_size=CHUNK_SIZE,
    )
    
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")

    print(f"✔ All done – results saved in: {FINAL_TSV}")

