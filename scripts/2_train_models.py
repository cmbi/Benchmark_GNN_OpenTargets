#!/usr/bin/env python3
"""
2_train_models.py  –  KG-Bench: Model Training
===============================================
Trains all six GNN architectures (GCN, GraphSAGE, TransformerConv, GAT, GIN, RGCN)
with early stopping, validation-loss-based model selection, and comprehensive logging.

Inputs  (from 1_create_graph.py):
  {timestamp}_graph.pt          – PyG Data object
  processed_data/mappings/all_mappings_{timestamp}.pkl

Outputs (saved to results_path/):
  {model_name}_L{l}_H{h}_best_model_{timestamp}.pt  – best weights per config
  training_summary_{timestamp}.json                  – metrics + paths
  training_report_{timestamp}.txt                    – human-readable summary
  {model_name}_training_curves_{timestamp}.png        – loss / AUC plots

Usage:
  python scripts/2_train_models.py path/to/graph.pt --mappings path/to/mappings.pkl
  python scripts/2_train_models.py path/to/graph.pt   # auto-locates latest mappings
"""

import os
import json
import pickle
import random
import datetime as dt
import itertools
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, GATConv, GINConv, RGCNConv


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_full_reproducibility(seed: int = 42):
    set_seed(seed)
    torch.use_deterministic_algorithms(True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_PATH = "results/"

def load_config(path: str = "config.json") -> dict:
    defaults = {
        "num_epochs": 1000,
        "patience":   10,
        "learning_rate": 0.0005,
        "dropout_rate":  0.5,
        "num_layers_list": [3],
        "hidden_dim_list": [16],
        "results_path": RESULTS_PATH,
        "negative_sampling_approach": "random",
    }
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS  (identical to main_script)
# ─────────────────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.conv1      = GCNConv(in_channels, hidden_channels)
        self.conv_list  = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers-1)])
        self.ln         = nn.LayerNorm(hidden_channels)
        self.dropout    = nn.Dropout(dropout_rate)
        self.final_layer= nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index)))
        x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = self.ln(conv(x, edge_index))
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


class SAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.conv1      = SAGEConv(in_channels, hidden_channels)
        self.conv_list  = nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers-1)])
        self.ln         = nn.LayerNorm(hidden_channels)
        self.dropout    = nn.Dropout(dropout_rate)
        self.final_layer= nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index)))
        x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = self.ln(conv(x, edge_index))
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


class TransformerModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers      = num_layers
        self.attention_weights = []
        self.conv1           = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv_list       = nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False)
             for _ in range(num_layers-1)])
        self.ln              = nn.LayerNorm(hidden_channels)
        self.dropout         = nn.Dropout(dropout_rate)
        self.final_layer     = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
        self.attention_weights = []
        if return_attention_weights:
            x, (ei, attn) = self.conv1(x, edge_index, return_attention_weights=True)
            self.attention_weights.append((ei, attn))
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(self.ln(x)); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            if return_attention_weights:
                x, (ei, attn) = conv(x, edge_index, return_attention_weights=True)
                self.attention_weights.append((ei, attn))
            else:
                x = conv(x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        x = self.final_layer(x)
        if return_attention_weights:
            return x, self.attention_weights
        return x


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5, heads=4):
        super().__init__()
        self.num_layers      = num_layers
        self.attention_weights = []
        self.conv1           = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv_list       = nn.ModuleList(
            [GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
             for _ in range(num_layers-1)])
        self.ln              = nn.LayerNorm(hidden_channels)
        self.dropout         = nn.Dropout(dropout_rate)
        self.final_layer     = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
        self.attention_weights = []
        if return_attention_weights:
            x, (ei, attn) = self.conv1(x, edge_index, return_attention_weights=True)
            self.attention_weights.append((ei, attn))
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(self.ln(x)); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            if return_attention_weights:
                x, (ei, attn) = conv(x, edge_index, return_attention_weights=True)
                self.attention_weights.append((ei, attn))
            else:
                x = conv(x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        x = self.final_layer(x)
        if return_attention_weights:
            return x, self.attention_weights
        return x


class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels))
        self.conv1     = GINConv(mlp1)
        self.conv_list = nn.ModuleList()
        for _ in range(num_layers-1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels))
            self.conv_list.append(GINConv(mlp))
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index))); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = self.ln(conv(x, edge_index))
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


class RGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5, num_relations=6):
        super().__init__()
        self.num_layers = num_layers
        self.conv1      = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv_list  = nn.ModuleList(
            [RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
             for _ in range(num_layers-1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index, edge_type))); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = self.ln(conv(x, edge_index, edge_type))
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


MODEL_CLASSES = {
    "GCNModel":         GCNModel,
    "SAGEModel":        SAGEModel,
    "TransformerModel": TransformerModel,
    "GATModel":         GATModel,
    "GINModel":         GINModel,
    "RGCNModel":        RGCNModel,
}


def _is_rgcn(model):
    return isinstance(model, RGCNModel)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(model, optimizer, graph, pos_edge_index, neg_edge_index,
                    val_edge_tensor, val_label_tensor, num_epochs, patience,
                    results_path, config_name, timestamp):
    """
    Train model with validation-loss early stopping.
    Returns (best_threshold, best_model_path, history_dict).
    """
    best_val_loss  = float("inf")
    best_threshold = 0.5
    counter        = 0
    loss_fn        = nn.BCEWithLogitsLoss()
    model_path     = f"{results_path}{config_name}_best_model_{timestamp}.pt"

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in tqdm(range(num_epochs), desc=f"Training {config_name}"):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        if _is_rgcn(model):
            z = model(graph.x.float(), graph.edge_index, graph.edge_type)
        else:
            z = model(graph.x.float(), graph.edge_index)

        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        pos_loss   = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss   = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss       = pos_loss + neg_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history["train_loss"].append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            if _is_rgcn(model):
                z = model(graph.x.float(), graph.edge_index, graph.edge_type)
            else:
                z = model(graph.x.float(), graph.edge_index)

            val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
            val_loss   = loss_fn(val_scores, val_label_tensor.float())
            val_probs  = torch.sigmoid(val_scores).cpu().numpy()
            val_threshold = val_probs.mean()

        history["val_loss"].append(val_loss.item())
        try:
            val_auc = roc_auc_score(val_label_tensor.cpu().numpy(), val_probs)
        except ValueError:
            val_auc = 0.5
        history["val_auc"].append(val_auc)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss.item()
            best_threshold = float(val_threshold)
            counter        = 0
            torch.save(model.state_dict(), model_path)
            print(f"  Epoch {epoch+1:4d}: val_loss={best_val_loss:.4f}  "
                  f"val_auc={val_auc:.4f}  threshold={best_threshold:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return best_threshold, model_path, history


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, graph, edge_tensor, label_tensor, threshold, batch_size=1000):
    """Return (probs, preds, metrics_dict)."""
    model.eval()
    probs_list = []
    with torch.no_grad():
        if _is_rgcn(model):
            z = model(graph.x.float(), graph.edge_index, graph.edge_type)
        else:
            z = model(graph.x.float(), graph.edge_index)

        for start in range(0, len(edge_tensor), batch_size):
            end   = min(start + batch_size, len(edge_tensor))
            batch = edge_tensor[start:end]
            s     = (z[batch[:, 0]] * z[batch[:, 1]]).sum(dim=-1)
            probs_list.append(torch.sigmoid(s).cpu().numpy())

    probs = np.concatenate(probs_list)
    preds = (probs >= threshold).astype(float)
    labels = label_tensor.cpu().numpy()

    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()

    precision   = TP / (TP + FP + 1e-10)
    recall      = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    f1          = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy    = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    auc_score   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    apr_score   = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    metrics = {
        "auc": float(auc_score), "apr": float(apr_score),
        "f1": float(f1), "accuracy": float(accuracy),
        "precision": float(precision), "recall": float(recall),
        "sensitivity": float(recall), "specificity": float(specificity),
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
    }
    return probs, preds, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(config_name, history, results_path, timestamp):
    epochs     = range(1, len(history["train_loss"]) + 1)
    val_epochs = range(1, len(history["val_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(epochs,     history["train_loss"], alpha=0.7, label="Train")
    axes[0].set_title(f"{config_name} – Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, history["val_loss"],   color="orange", label="Val")
    axes[1].set_title(f"{config_name} – Validation Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(val_epochs, history["val_auc"],    color="green",  label="Val AUC")
    axes[2].set_title(f"{config_name} – Validation AUC"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{results_path}{config_name}_training_curves_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(graph_path: str, mappings_path: str,
                     results_path: str = RESULTS_PATH,
                     cfg: dict = None):
    """Train all 6 GNN architectures with configurations from cfg."""
    cfg = cfg or load_config()
    os.makedirs(results_path, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    enable_full_reproducibility(42)

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f"\nLoading graph from {graph_path} …")
    graph = torch.load(graph_path, map_location=device)
    graph.x = graph.x.float()
    print(graph)

    # ── Load mappings ─────────────────────────────────────────────────────────
    print(f"Loading mappings from {mappings_path} …")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    # ── Extract pos/neg training edges ────────────────────────────────────────
    # pos_edge_index stored as [2, N] tensor in graph (from 1_create_graph.py)
    if hasattr(graph, "pos_edge_index") and graph.pos_edge_index.numel() > 0:
        pos_edge_index = graph.pos_edge_index.to(device)
        neg_edge_index = graph.neg_edge_index.to(device)
        print(f"Using stored training edges: pos={pos_edge_index.size(1):,}, "
              f"neg={neg_edge_index.size(1):,}")
    else:
        raise RuntimeError(
            "Graph is missing pos_edge_index / neg_edge_index. "
            "Re-run 1_create_graph.py to regenerate."
        )

    # Validation tensors
    val_edge_tensor   = graph.val_edge_index.to(device)
    val_label_tensor  = graph.val_edge_label.to(device)

    print(f"Validation set: {val_edge_tensor.size(0):,} samples")

    # ── Configuration grid ────────────────────────────────────────────────────
    num_layers_list = cfg.get("num_layers_list", [3])
    hidden_dim_list = cfg.get("hidden_dim_list", [16])
    config_combinations = list(itertools.product(num_layers_list, hidden_dim_list))
    num_epochs  = cfg.get("num_epochs", 1000)
    patience    = cfg.get("patience",   10)
    lr          = cfg.get("learning_rate", 0.0005)
    dropout     = cfg.get("dropout_rate",  0.5)
    num_relations = int(graph.metadata.get("num_relations", 6)) if hasattr(graph, "metadata") else 6

    print(f"\nConfigurations: {config_combinations}")
    print(f"Models: {list(MODEL_CLASSES.keys())}")
    print(f"Total runs: {len(config_combinations) * len(MODEL_CLASSES)}")

    # ── Train loop ────────────────────────────────────────────────────────────
    all_results   = []
    best_models   = {}   # {arch_name: {model, config, auc, metrics, model_path}}

    for model_name, model_class in MODEL_CLASSES.items():
        print(f"\n{'='*80}")
        print(f"ARCHITECTURE: {model_name}")
        print(f"{'='*80}")

        best_auc_arch = -1

        for (num_layers, hidden_dim) in config_combinations:
            config_name = f"{model_name}_L{num_layers}_H{hidden_dim}"
            torch.cuda.empty_cache()
            set_seed(42)

            print(f"\n--- {config_name} ---")

            # Build model
            kwargs = dict(
                in_channels     = graph.x.size(1),
                hidden_channels = hidden_dim,
                out_channels    = hidden_dim,
                num_layers      = num_layers,
                dropout_rate    = dropout,
            )
            if model_name == "RGCNModel":
                kwargs["num_relations"] = num_relations
            model = model_class(**kwargs).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train
            best_threshold, model_path, history = train_one_model(
                model, optimizer, graph,
                pos_edge_index, neg_edge_index,
                val_edge_tensor, val_label_tensor,
                num_epochs, patience,
                results_path, config_name, timestamp
            )

            # Reload best weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # Evaluate on validation set
            val_probs, val_preds, val_metrics = evaluate_model(
                model, graph, val_edge_tensor, val_label_tensor, best_threshold
            )
            print(f"  Val AUC={val_metrics['auc']:.3f}  "
                  f"APR={val_metrics['apr']:.3f}  F1={val_metrics['f1']:.3f}")

            # Training curves
            save_training_curves(config_name, history, results_path, timestamp)

            result_entry = {
                "name":       config_name,
                "base_model": model_name,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "threshold":  best_threshold,
                "model_path": model_path,
                "val_metrics": val_metrics,
            }
            all_results.append(result_entry)

            if val_metrics["auc"] > best_auc_arch:
                best_auc_arch = val_metrics["auc"]
                best_models[model_name] = {
                    "model":      model,
                    "config":     config_name,
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "auc":        best_auc_arch,
                    "threshold":  best_threshold,
                    "model_path": model_path,
                    "metrics":    val_metrics,
                }

        print(f"\n  {model_name} best: "
              f"{best_models[model_name]['config']}  "
              f"(val AUC {best_auc_arch:.3f})")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "timestamp":     timestamp,
        "graph_path":    graph_path,
        "mappings_path": mappings_path,
        "device":        str(device),
        "num_epochs":    num_epochs,
        "patience":      patience,
        "learning_rate": lr,
        "dropout_rate":  dropout,
        "configurations": config_combinations,
        "all_results": [
            {k: v for k, v in r.items() if k != "model"} for r in all_results
        ],
        "best_models": {
            arch: {
                "config":     info["config"],
                "num_layers": info["num_layers"],
                "hidden_dim": info["hidden_dim"],
                "auc":        info["auc"],
                "threshold":  info["threshold"],
                "model_path": info["model_path"],
                "metrics":    info["metrics"],
            }
            for arch, info in best_models.items()
        },
    }

    summary_json = f"{results_path}training_summary_{timestamp}.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Text report
    report_txt = f"{results_path}training_report_{timestamp}.txt"
    with open(report_txt, "w") as f:
        f.write("KG-BENCH: MODEL TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Completed: {timestamp}\n")
        f.write(f"Device:    {device}\n\n")
        f.write("BEST CONFIGURATION PER ARCHITECTURE:\n")
        f.write("-"*60 + "\n")
        for arch, info in best_models.items():
            f.write(f"\n{arch}:\n")
            f.write(f"  Config:     {info['config']}\n")
            f.write(f"  Layers:     {info['num_layers']}\n")
            f.write(f"  Hidden:     {info['hidden_dim']}\n")
            f.write(f"  Val AUC:    {info['auc']:.4f}\n")
            f.write(f"  Val APR:    {info['metrics']['apr']:.4f}\n")
            f.write(f"  Model path: {info['model_path']}\n")

        f.write("\n\nALL CONFIGURATIONS:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Config':<35} {'AUC':>8} {'APR':>8} {'F1':>8}\n")
        for r in sorted(all_results, key=lambda x: x["val_metrics"]["auc"], reverse=True):
            m = r["val_metrics"]
            f.write(f"{r['name']:<35} {m['auc']:>8.4f} {m['apr']:>8.4f} {m['f1']:>8.4f}\n")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Summary JSON: {summary_json}")
    print(f"Text report:  {report_txt}")
    print(f"\nBest models by validation AUC:")
    for arch, info in best_models.items():
        print(f"  {arch}: {info['config']}  AUC={info['auc']:.3f}")

    print(f"\nNext step:")
    print(f"  python scripts/3_test_evaluate.py {graph_path} {results_path} "
          f"--mappings {mappings_path} --summary {summary_json}")

    return best_models, all_results, summary


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_mappings(processed_path="processed_data/mappings/"):
    """Auto-locate the most recent mappings pickle."""
    candidates = sorted(
        [f for f in os.listdir(processed_path) if f.endswith(".pkl")],
        reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No mappings .pkl found in {processed_path}")
    return os.path.join(processed_path, candidates[0])


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN models for drug–disease prediction (KG-Bench)"
    )
    parser.add_argument("graph_path",
                        help="Path to graph .pt file from 1_create_graph.py")
    parser.add_argument("--mappings", default=None,
                        help="Path to mappings .pkl (auto-detected if omitted)")
    parser.add_argument("--results-path", default=RESULTS_PATH,
                        help="Output directory for models and reports")
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph not found: {args.graph_path}")

    mappings_path = args.mappings or _find_latest_mappings()
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings not found: {mappings_path}")

    cfg = load_config(args.config)
    cfg["results_path"] = args.results_path

    train_all_models(args.graph_path, mappings_path, args.results_path, cfg)


if __name__ == "__main__":
    main()
