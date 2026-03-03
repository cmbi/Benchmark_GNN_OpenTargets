#!/usr/bin/env python3
"""
3_test_evaluate.py  –  KG-Bench: Model Testing & Evaluation
=============================================================
Tests all six trained GNN architectures on the temporal test set,
generates publication-quality figures with bootstrap confidence intervals,
and exports repurposing candidate predictions (FP scan).

Inputs (from previous steps):
  {timestamp}_graph.pt              – PyG Data object
  processed_data/mappings/all_mappings_{timestamp}.pkl
  training_summary_{timestamp}.json – from 2_train_models.py

Outputs (saved to results_path/):
  test_results_summary_{timestamp}.csv
  test_evaluation_report_{timestamp}.txt
  ROC_curves_6models_{timestamp}.png
  PR_curves_6models_{timestamp}.png
  Combined_ROC_PR_6models_{timestamp}.png
  {config_name}_confusion_matrix_{timestamp}.png
  {config_name}_FP_AllPairs_{timestamp}.csv   (TransformerModel only)
  {config_name}_FP_AllPairs_{timestamp}.parquet
  IMBALANCE_ANALYSIS_{timestamp}.txt

Usage:
  python scripts/3_test_evaluate.py path/to/graph.pt path/to/results_dir/
      --mappings path/to/mappings.pkl
      --summary  path/to/training_summary.json
"""

import os
import csv
import json
import pickle
import random
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc,
    confusion_matrix,
)
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


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS  (identical to 2_train_models.py / main_script)
# ─────────────────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.conv1      = GCNConv(in_channels, hidden_channels)
        self.conv_list  = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers-1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index))); x = self.dropout(x)
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
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
        x = F.relu(self.ln(self.conv1(x, edge_index))); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = self.ln(conv(x, edge_index))
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


class TransformerModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers        = num_layers
        self.attention_weights = []
        self.conv1     = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv_list = nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False)
             for _ in range(num_layers-1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

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
        self.num_layers        = num_layers
        self.attention_weights = []
        self.conv1     = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv_list = nn.ModuleList(
            [GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
             for _ in range(num_layers-1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

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
        mlp1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                              nn.Linear(hidden_channels, hidden_channels))
        self.conv1     = GINConv(mlp1)
        self.conv_list = nn.ModuleList()
        for _ in range(num_layers-1):
            mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                                 nn.Linear(hidden_channels, hidden_channels))
            self.conv_list.append(GINConv(mlp))
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
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

    def forward(self, x, edge_index, return_attention_weights=False, edge_type=None):
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

# Color scheme (matches main_script)
COLORS_MAP = {
    "Transformer": "green", "SAGE": "purple", "GCN": "gray",
    "GIN": "orange",        "GAT": "blue",    "RGCN": "red",
}


def _is_rgcn(model):
    return isinstance(model, RGCNModel)


def _base_model_name(full_name: str) -> str:
    return full_name.split("_")[0].replace("Model", "")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD TRAINED MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_models(summary: dict, graph, device):
    """Reconstruct trained models from saved state dicts."""
    models = {}
    in_channels = graph.x.size(1)
    num_relations = int(graph.metadata.get("num_relations", 6)) if hasattr(graph, "metadata") else 6

    for arch, info in summary.get("best_models", {}).items():
        model_class = MODEL_CLASSES.get(arch)
        if model_class is None:
            print(f"  WARNING: Unknown model class '{arch}', skipping.")
            continue

        kwargs = dict(
            in_channels     = in_channels,
            hidden_channels = info["hidden_dim"],
            out_channels    = info["hidden_dim"],
            num_layers      = info["num_layers"],
            dropout_rate    = 0.5,
        )
        if arch == "RGCNModel":
            kwargs["num_relations"] = num_relations

        model = model_class(**kwargs).to(device)

        model_path = info["model_path"]
        if not os.path.exists(model_path):
            print(f"  WARNING: Model file not found: {model_path}")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        models[arch] = {
            "model":     model,
            "config":    info["config"],
            "threshold": info["threshold"],
            "num_layers": info["num_layers"],
            "hidden_dim": info["hidden_dim"],
            "model_path": model_path,
        }
        print(f"  Loaded {info['config']} (threshold={info['threshold']:.4f})")

    return models


# ─────────────────────────────────────────────────────────────────────────────
#  BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(labels, probs, metric="auc", n_bootstraps=1000, confidence=0.95, seed=42):
    """Return (point_estimate, ci_lower, ci_upper)."""
    rng = np.random.RandomState(seed)
    n   = len(labels)

    if metric == "auc":
        fn = roc_auc_score
    elif metric == "apr":
        fn = average_precision_score
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    point = fn(labels, probs)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        scores.append(fn(labels[idx], probs[idx]))

    alpha = 1 - confidence
    ci_lower = np.percentile(scores, 100 * alpha / 2)
    ci_upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(point), float(ci_lower), float(ci_upper)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def test_model(model, graph, test_edge_tensor, test_label_tensor,
               threshold, batch_size=1000):
    """Return (probs, preds, metrics)."""
    model.eval()
    probs_list = []

    with torch.no_grad():
        if _is_rgcn(model):
            z = model(graph.x.float(), graph.edge_index, edge_type=graph.edge_type)
        else:
            z = model(graph.x.float(), graph.edge_index)

        for start in range(0, len(test_edge_tensor), batch_size):
            end   = min(start + batch_size, len(test_edge_tensor))
            batch = test_edge_tensor[start:end]
            s     = (z[batch[:, 0]] * z[batch[:, 1]]).sum(dim=-1)
            probs_list.append(torch.sigmoid(s).cpu().numpy())

    probs  = np.concatenate(probs_list)
    preds  = (probs >= threshold).astype(float)
    labels = test_label_tensor.cpu().numpy()

    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()

    precision   = TP / (TP + FP + 1e-10)
    recall      = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    f1          = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy    = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    ppv = precision
    npv = TN / (TN + FN + 1e-10)

    auc_val = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    apr_val = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    metrics = {
        "auc": float(auc_val), "apr": float(apr_val),
        "f1": float(f1), "accuracy": float(accuracy),
        "precision": float(precision), "recall": float(recall),
        "sensitivity": float(recall), "specificity": float(specificity),
        "ppv": float(ppv), "npv": float(npv),
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
    }
    return probs, preds, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  FALSE-POSITIVE SCAN  (full drug × disease enumeration, mirrors main_script)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save_false_positives(model, graph, threshold,
                                     results_path, config_name, timestamp,
                                     approved_drugs_list_name, disease_list_name,
                                     disease_offset, known_positive_edges_set,
                                     batch_size=50_000):
    """
    Score every (drug, disease) pair not in known_positive_edges_set.
    Writes all predicted-positive pairs to CSV + Parquet (streamed, no memory cap).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    device   = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        if _is_rgcn(model):
            z = model(graph.x.float(), graph.edge_index, edge_type=graph.edge_type)
        else:
            z = model(graph.x.float(), graph.edge_index)
    z = z.cpu()

    num_drugs    = len(approved_drugs_list_name)
    num_diseases = len(disease_list_name)
    total_pairs  = num_drugs * num_diseases

    print(f"\n{'='*80}")
    print(f"FULL CANDIDATE SCAN — {config_name}")
    print(f"  Drugs            : {num_drugs:,}")
    print(f"  Diseases         : {num_diseases:,}")
    print(f"  Total pairs      : {total_pairs:,}")
    print(f"  Known positives  : {len(known_positive_edges_set):,}")
    print(f"  Batch size       : {batch_size:,}")
    print(f"{'='*80}")

    csv_path     = f"{results_path}{config_name}_FP_AllPairs_{timestamp}.csv"
    parquet_path = f"{results_path}{config_name}_FP_AllPairs_{timestamp}.parquet"

    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Drug", "Disease", "Probability"])

    disease_node_indices = list(range(disease_offset, disease_offset + num_diseases))
    drug_node_indices    = list(range(num_drugs))

    def _pair_gen():
        for d_i in drug_node_indices:
            for dis_i in disease_node_indices:
                yield d_i, dis_i

    gen           = _pair_gen()
    done          = False
    batch_num     = 0
    total_written = 0
    parquet_chunks = []
    parquet_rows   = []
    FLUSH_EVERY    = 20

    while not done:
        bd, bdi = [], []
        for _ in range(batch_size):
            try:
                d, dis = next(gen)
                bd.append(d); bdi.append(dis)
            except StopIteration:
                done = True; break

        if not bd:
            break
        batch_num += 1

        with torch.no_grad():
            bt  = torch.tensor(bd,  dtype=torch.long)
            bdt = torch.tensor(bdi, dtype=torch.long)
            scores = (z[bt] * z[bdt]).sum(dim=-1)
            probs  = torch.sigmoid(scores).numpy()

        preds = probs >= threshold
        for i in range(len(bd)):
            if not preds[i]:
                continue
            pair = (bd[i], bdi[i])
            if pair in known_positive_edges_set:
                continue
            prob         = float(probs[i])
            drug_name    = approved_drugs_list_name[bd[i]]
            disease_name = disease_list_name[bdi[i] - disease_offset]
            csv_writer.writerow([drug_name, disease_name, f"{prob:.8f}"])
            parquet_rows.append({"Drug": drug_name, "Disease": disease_name, "Probability": prob})
            total_written += 1

        if batch_num % FLUSH_EVERY == 0 or done:
            if parquet_rows:
                chunk = pa.table({
                    "Drug":        pa.array([r["Drug"]        for r in parquet_rows]),
                    "Disease":     pa.array([r["Disease"]     for r in parquet_rows]),
                    "Probability": pa.array([r["Probability"] for r in parquet_rows],
                                            type=pa.float32()),
                })
                parquet_chunks.append(chunk)
                parquet_rows = []

        if batch_num % 100 == 0:
            print(f"  Scored {batch_num*batch_size:,} pairs | candidates: {total_written:,}")

    csv_file.close()

    if parquet_chunks:
        pq.write_table(pa.concat_tables(parquet_chunks), parquet_path, compression="snappy")

    print(f"\n  Total FP candidates : {total_written:,}")
    print(f"  CSV  → {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB)")

    # Top-30 preview
    if total_written > 0:
        df_peek = pd.read_csv(csv_path).sort_values("Probability", ascending=False).head(30)
        print("\n  Top 30 repurposing candidates:")
        for rank, row in enumerate(df_peek.itertuples(), 1):
            print(f"    {rank:2d}. {row.Drug} → {row.Disease}  ({row.Probability:.4f})")

    return csv_path, parquet_path


# ─────────────────────────────────────────────────────────────────────────────
#  IMBALANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_imbalanced_test(test_pos_set, not_linked_pool, ratio, seed=42):
    """Create (edges [N,2], labels [N]) with given positive:negative ratio."""
    random.seed(seed)
    true_pairs  = list(test_pos_set)
    num_neg     = min(len(true_pairs) * ratio, len(not_linked_pool))
    false_pairs = random.sample(not_linked_pool, num_neg)
    edges  = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
    labels = torch.tensor([1]*len(true_pairs) + [0]*len(false_pairs), dtype=torch.long)
    print(f"  Imbalance 1:{ratio} | pos={len(true_pairs)}, neg={len(false_pairs)}, "
          f"prevalence={len(true_pairs)/(len(true_pairs)+len(false_pairs)):.3f}")
    return edges, labels


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results_list, results_path, timestamp, show_ci=True):
    plt.figure(figsize=(10, 10))
    sorted_res = sorted(results_list, key=lambda x: x["metrics"]["auc"], reverse=True)
    for res in sorted_res:
        base = _base_model_name(res["name"])
        color = COLORS_MAP.get(base, "black")
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        if show_ci:
            _, lo, hi = bootstrap_ci(res["labels"], res["probs"], metric="auc")
            label = f"{base} (AUC={res['metrics']['auc']:.3f} [{lo:.3f}–{hi:.3f}])"
        else:
            label = f"{base} (AUC={res['metrics']['auc']:.3f})"
        plt.plot(fpr, tpr, color=color, lw=2, label=label)

    plt.plot([0,1],[0,1], "--", color="gray", lw=1.5, alpha=0.7, label="Random")
    plt.xlabel("False Positive Rate", fontsize=18, fontweight="bold")
    plt.ylabel("True Positive Rate",  fontsize=18, fontweight="bold")
    plt.tick_params(labelsize=16)
    plt.legend(loc="lower right", fontsize=13, shadow=True)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{results_path}ROC_curves_6models_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_pr_curves(results_list, results_path, timestamp, show_ci=True):
    plt.figure(figsize=(10, 10))
    sorted_res = sorted(results_list,
                        key=lambda x: average_precision_score(x["labels"], x["probs"]),
                        reverse=True)
    for res in sorted_res:
        base = _base_model_name(res["name"])
        color = COLORS_MAP.get(base, "black")
        precision, recall, _ = precision_recall_curve(res["labels"], res["probs"])
        ap = average_precision_score(res["labels"], res["probs"])
        if show_ci:
            _, lo, hi = bootstrap_ci(res["labels"], res["probs"], metric="apr")
            label = f"{base} (AP={ap:.3f} [{lo:.3f}–{hi:.3f}])"
        else:
            label = f"{base} (AP={ap:.3f})"
        plt.plot(recall, precision, color=color, lw=2, label=label)

    prevalence = np.mean(results_list[0]["labels"]) if results_list else 0.5
    plt.plot([0,1],[prevalence,prevalence], "--", color="gray", lw=1.5, alpha=0.7,
             label="Prevalence")
    plt.xlabel("Recall",    fontsize=18, fontweight="bold")
    plt.ylabel("Precision", fontsize=18, fontweight="bold")
    plt.tick_params(labelsize=16)
    plt.xlim([0, 1]); plt.ylim([0.5, 1.05])
    plt.legend(loc="best", fontsize=13, shadow=True)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = f"{results_path}PR_curves_6models_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_combined(results_list, results_path, timestamp, show_ci=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    sorted_auc = sorted(results_list, key=lambda x: x["metrics"]["auc"], reverse=True)
    for res in sorted_auc:
        base  = _base_model_name(res["name"])
        color = COLORS_MAP.get(base, "black")
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        if show_ci:
            _, lo, hi = bootstrap_ci(res["labels"], res["probs"], metric="auc")
            label = f"{base} (AUC={res['metrics']['auc']:.3f} [{lo:.3f}–{hi:.3f}])"
        else:
            label = f"{base} (AUC={res['metrics']['auc']:.3f})"
        ax1.plot(fpr, tpr, color=color, lw=2, label=label)
    ax1.plot([0,1],[0,1], "--", color="gray", lw=1.5, alpha=0.7)
    ax1.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold")
    ax1.set_ylabel("True Positive Rate",  fontsize=18, fontweight="bold")
    ax1.tick_params(labelsize=16); ax1.legend(loc="lower right", fontsize=13, shadow=True)
    ax1.grid(True, alpha=0.3)

    sorted_ap = sorted(results_list,
                       key=lambda x: average_precision_score(x["labels"], x["probs"]),
                       reverse=True)
    for res in sorted_ap:
        base  = _base_model_name(res["name"])
        color = COLORS_MAP.get(base, "black")
        precision, recall, _ = precision_recall_curve(res["labels"], res["probs"])
        ap = average_precision_score(res["labels"], res["probs"])
        if show_ci:
            _, lo, hi = bootstrap_ci(res["labels"], res["probs"], metric="apr")
            label = f"{base} (AP={ap:.3f} [{lo:.3f}–{hi:.3f}])"
        else:
            label = f"{base} (AP={ap:.3f})"
        ax2.plot(recall, precision, color=color, lw=2, label=label)
    prevalence = np.mean(results_list[0]["labels"]) if results_list else 0.5
    ax2.plot([0,1],[prevalence,prevalence], "--", color="gray", lw=1.5, alpha=0.7)
    ax2.set_xlabel("Recall",    fontsize=18, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=18, fontweight="bold")
    ax2.tick_params(labelsize=16); ax2.set_xlim([0,1]); ax2.set_ylim([0.5, 1.05])
    ax2.legend(loc="best", fontsize=13, shadow=True); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{results_path}Combined_ROC_PR_6models_{timestamp}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(config_name, labels, preds, results_path, timestamp):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {config_name}")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.xticks([0.5,1.5], ["Negative","Positive"])
    plt.yticks([0.5,1.5], ["Negative","Positive"], rotation=0)
    plt.tight_layout()
    path = f"{results_path}{config_name}_confusion_matrix_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return cm


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(graph_path: str, results_path: str,
                   mappings_path: str, summary_path: str,
                   export_fp: bool = True,
                   imbalance_ratios: list = None):
    if imbalance_ratios is None:
        imbalance_ratios = [1, 10, 100]

    os.makedirs(results_path, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f"Loading graph from {graph_path} …")
    graph = torch.load(graph_path, map_location=device)
    graph.x = graph.x.float()
    print(graph)

    # ── Load mappings ─────────────────────────────────────────────────────────
    print(f"Loading mappings from {mappings_path} …")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    approved_drugs_list_name = mappings["approved_drugs_list_name"]
    disease_list_name        = mappings["disease_list_name"]
    disease_offset           = mappings["disease_offset"]

    # ── Load training summary ─────────────────────────────────────────────────
    print(f"Loading training summary from {summary_path} …")
    with open(summary_path) as f:
        summary = json.load(f)

    # ── Load trained models ───────────────────────────────────────────────────
    print("\nLoading trained models …")
    trained_models = load_trained_models(summary, graph, device)

    if not trained_models:
        raise RuntimeError("No trained models could be loaded. Check model paths in summary.")

    # ── Balanced (1:1) test set ───────────────────────────────────────────────
    test_edge_tensor  = graph.test_edge_index.to(device)
    test_label_tensor = graph.test_edge_label.to(device)

    print(f"\nTest set: {test_edge_tensor.size(0):,} samples "
          f"({int(test_label_tensor.sum())} positive)")

    # ── Known positives for FP scan ───────────────────────────────────────────
    train_pos_edges = graph.pos_edge_index.t().tolist() if hasattr(graph, "pos_edge_index") else []
    val_pos_edges   = graph.val_pos_edges.tolist()  if hasattr(graph, "val_pos_edges")  else []
    test_pos_edges  = graph.test_pos_edges.tolist() if hasattr(graph, "test_pos_edges") else []
    known_positives = (
        set(map(tuple, train_pos_edges)) |
        set(map(tuple, val_pos_edges))   |
        set(map(tuple, test_pos_edges))
    )
    print(f"Known positive edges (train+val+test): {len(known_positives):,}")

    # ── Negative pool for imbalance analysis ─────────────────────────────────
    test_pos_set = set(map(tuple, test_pos_edges)) if test_pos_edges else (
        set(zip(
            test_edge_tensor[test_label_tensor == 1, 0].cpu().tolist(),
            test_edge_tensor[test_label_tensor == 1, 1].cpu().tolist()
        ))
    )
    # All drug×disease combinations not in any positive set
    all_drug_disease = [
        (mappings["drug_key_mapping"][d], mappings["disease_key_mapping"][dis])
        for d in mappings["approved_drugs_list"]
        for dis in mappings["disease_list"]
    ]
    not_linked_pool = list(set(all_drug_disease) - known_positives)
    random.shuffle(not_linked_pool)
    print(f"Negative pool size: {len(not_linked_pool):,}")

    # ── Test all models (1:1 balanced) ────────────────────────────────────────
    print("\n" + "="*80)
    print("TESTING ALL MODELS (balanced 1:1)")
    print("="*80)

    all_results = []
    model_result_dict = {}

    for arch, model_info in trained_models.items():
        model      = model_info["model"]
        config_nm  = model_info["config"]
        threshold  = model_info["threshold"]

        print(f"\n--- {config_nm} ---")

        probs, preds, metrics = test_model(
            model, graph, test_edge_tensor, test_label_tensor, threshold
        )

        cm = plot_confusion_matrix(config_nm,
                                   test_label_tensor.cpu().numpy(), preds,
                                   results_path, timestamp)

        result = {
            "name":    config_nm,
            "arch":    arch,
            "probs":   probs,
            "preds":   preds,
            "labels":  test_label_tensor.cpu().numpy(),
            "metrics": metrics,
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),
        }
        all_results.append(result)
        model_result_dict[arch] = result

        print(f"  AUC={metrics['auc']:.4f}  APR={metrics['apr']:.4f}  "
              f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    # ── FP export for TransformerModel ────────────────────────────────────────
    if export_fp and "TransformerModel" in trained_models:
        print("\n" + "="*80)
        print("EXPORTING REPURPOSING CANDIDATES (TransformerModel)")
        print("="*80)
        model_info = trained_models["TransformerModel"]
        extract_and_save_false_positives(
            model      = model_info["model"],
            graph      = graph,
            threshold  = model_info["threshold"],
            results_path = results_path,
            config_name  = model_info["config"],
            timestamp    = timestamp,
            approved_drugs_list_name = approved_drugs_list_name,
            disease_list_name        = disease_list_name,
            disease_offset           = disease_offset,
            known_positive_edges_set = known_positives,
            batch_size = 50_000,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating publication-quality plots …")
    plot_roc_curves(all_results, results_path, timestamp)
    plot_pr_curves( all_results, results_path, timestamp)
    plot_combined(  all_results, results_path, timestamp)

    # ── Imbalance analysis ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("IMBALANCE ANALYSIS")
    print("="*80)
    imbalance_results = {}

    for ratio in imbalance_ratios:
        print(f"\n  Imbalance ratio 1:{ratio}")
        imb_edges, imb_labels = build_imbalanced_test(
            test_pos_set, not_linked_pool, ratio
        )
        imb_edges  = imb_edges.to(device)
        imb_labels = imb_labels.to(device)

        ratio_results = []
        for arch, model_info in trained_models.items():
            probs_i, preds_i, metrics_i = test_model(
                model_info["model"], graph, imb_edges, imb_labels,
                model_info["threshold"]
            )
            ratio_results.append({
                "model":   model_info["config"],
                "metrics": metrics_i,
            })
            print(f"    {model_info['config']:35s}  "
                  f"AUC={metrics_i['auc']:.3f}  APR={metrics_i['apr']:.3f}  "
                  f"F1={metrics_i['f1']:.3f}")

        imbalance_results[ratio] = ratio_results

    # Save imbalance analysis
    imb_txt = f"{results_path}IMBALANCE_ANALYSIS_{timestamp}.txt"
    with open(imb_txt, "w") as f:
        f.write("="*80 + "\n")
        f.write("KG-BENCH: IMBALANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write("Addresses: 'How do GNN metrics change under class imbalance?'\n\n")

        for ratio in imbalance_ratios:
            f.write(f"\nRatio 1:{ratio}\n"); f.write("-"*60 + "\n")
            f.write(f"{'Config':<35} {'AUC':>8} {'APR':>8} {'F1':>8}\n")
            for r in sorted(imbalance_results[ratio],
                             key=lambda x: x["metrics"]["auc"], reverse=True):
                m = r["metrics"]
                f.write(f"{r['model']:<35} {m['auc']:>8.3f} {m['apr']:>8.3f} {m['f1']:>8.3f}\n")
    print(f"\nImbalance analysis saved → {imb_txt}")

    # ── Save CSV summary ───────────────────────────────────────────────────────
    csv_path = f"{results_path}test_results_summary_{timestamp}.csv"
    rows = []
    for res in sorted(all_results, key=lambda x: x["metrics"]["auc"], reverse=True):
        m = res["metrics"]
        base = _base_model_name(res["name"])
        _, auc_lo, auc_hi = bootstrap_ci(res["labels"], res["probs"], metric="auc")
        _, apr_lo, apr_hi = bootstrap_ci(res["labels"], res["probs"], metric="apr")
        rows.append({
            "Model":          base,
            "Configuration":  res["name"],
            "AUC":            m["auc"],
            "AUC_CI_lower":   auc_lo,
            "AUC_CI_upper":   auc_hi,
            "APR":            m["apr"],
            "APR_CI_lower":   apr_lo,
            "APR_CI_upper":   apr_hi,
            "F1":             m["f1"],
            "Accuracy":       m["accuracy"],
            "Precision":      m["precision"],
            "Recall":         m["recall"],
            "Specificity":    m["specificity"],
            "Threshold":      res["threshold"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV summary → {csv_path}")

    # ── Text evaluation report ─────────────────────────────────────────────────
    report_path = f"{results_path}test_evaluation_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("KG-BENCH: TEST EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Graph     : {graph_path}\n")
        f.write(f"Mappings  : {mappings_path}\n")
        f.write(f"Device    : {device}\n\n")
        f.write(f"Test set  : {test_edge_tensor.size(0):,} samples "
                f"({int(test_label_tensor.sum())} pos)\n\n")

        f.write("MODEL PERFORMANCE (balanced 1:1 test set):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<35} {'AUC':>8} {'AUC 95%CI':>22} {'APR':>8} {'F1':>8}\n")
        f.write("-"*80 + "\n")
        for res in sorted(all_results, key=lambda x: x["metrics"]["auc"], reverse=True):
            m = res["metrics"]
            _, alo, ahi = bootstrap_ci(res["labels"], res["probs"], metric="auc")
            f.write(f"{res['name']:<35} {m['auc']:>8.4f} "
                    f"[{alo:.4f}–{ahi:.4f}]  {m['apr']:>8.4f} {m['f1']:>8.4f}\n")

        f.write("\n\nCONFUSION MATRICES:\n")
        f.write("-"*80 + "\n")
        for res in all_results:
            m = res["metrics"]
            f.write(f"\n{res['name']}:\n")
            f.write(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}\n")
            f.write(f"  Sensitivity={m['sensitivity']:.4f}  Specificity={m['specificity']:.4f}\n")
            f.write(f"  PPV={m['ppv']:.4f}  NPV={m['npv']:.4f}\n")

    print(f"Evaluation report → {report_path}")

    # ── Final summary to stdout ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n{'Model':<35} {'AUC':>8} {'APR':>8} {'F1':>8}")
    print("-"*60)
    for res in sorted(all_results, key=lambda x: x["metrics"]["auc"], reverse=True):
        m = res["metrics"]
        print(f"{res['name']:<35} {m['auc']:>8.4f} {m['apr']:>8.4f} {m['f1']:>8.4f}")

    return {
        "balanced_results":   all_results,
        "imbalance_results":  imbalance_results,
        "csv_path":           csv_path,
        "report_path":        report_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_file(directory, pattern):
    """Return the most recent file matching pattern in directory."""
    import glob
    files = sorted(glob.glob(os.path.join(directory, pattern)), reverse=True)
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Test and evaluate GNN models for drug–disease prediction (KG-Bench)"
    )
    parser.add_argument("graph_path",
                        help="Path to graph .pt file from 1_create_graph.py")
    parser.add_argument("models_path",
                        help="Directory containing trained models / results from 2_train_models.py")
    parser.add_argument("--mappings", default=None,
                        help="Path to mappings .pkl (auto-detected if omitted)")
    parser.add_argument("--summary",  default=None,
                        help="Path to training_summary.json (auto-detected if omitted)")
    parser.add_argument("--results-path", default=None,
                        help="Output directory (defaults to models_path)")
    parser.add_argument("--no-fp-export", action="store_true",
                        help="Skip repurposing candidate export")
    parser.add_argument("--imbalance-ratios", nargs="+", type=int,
                        default=[1, 10, 100],
                        help="Imbalance ratios to test (default: 1 10 100)")
    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph not found: {args.graph_path}")

    results_path = args.results_path or args.models_path
    os.makedirs(results_path, exist_ok=True)

    # Auto-detect mappings
    if args.mappings:
        mappings_path = args.mappings
    else:
        mappings_path = _find_latest_file("processed_data/mappings/", "all_mappings_*.pkl")
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings not found: {mappings_path}")

    # Auto-detect summary
    if args.summary:
        summary_path = args.summary
    else:
        summary_path = _find_latest_file(args.models_path, "training_summary_*.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Training summary not found: {summary_path}")

    run_evaluation(
        graph_path       = args.graph_path,
        results_path     = results_path,
        mappings_path    = mappings_path,
        summary_path     = summary_path,
        export_fp        = not args.no_fp_export,
        imbalance_ratios = args.imbalance_ratios,
    )


if __name__ == "__main__":
    main()
