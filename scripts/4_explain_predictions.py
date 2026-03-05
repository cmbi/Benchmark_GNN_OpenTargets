#!/usr/bin/env python3
"""
4_explainer.py  –  KG-Bench: GNNExplainer Attribution & Interactive Visualization
===================================================================================
Generates model-faithful explanations for drug-disease repurposing predictions
using GNNExplainer with a proper LinkPredictor wrapper, degree-adjusted importance
scoring, bootstrap confidence intervals, and an interactive D3.js HTML visualizer.

Inputs (from previous pipeline steps):
  {timestamp}_graph.pt                      – PyG Data object (from 1_create_graph.py)
  processed_data/mappings/all_mappings_{timestamp}.pkl
  training_summary_{timestamp}.json         – from 2_train_models.py
  {model}_FP_AllPairs_{timestamp}.csv       – FP candidates (from 3_test_evaluate.py)
                                              (auto-detected or supplied via --fp-csv)

Outputs (saved to results_path/):
  GNNExplainer_importance_{model}_{timestamp}.txt   – bootstrap CI report
  GNNExplainer_visualization_{model}_{timestamp}.html  – interactive D3 visualizer
  GNNExplainer_node_importance_{model}_{timestamp}.csv – per-node scores
  GNNExplainer_edge_importance_{model}_{timestamp}.csv – per-edge scores

Usage:
  python scripts/4_explainer.py path/to/graph.pt path/to/results_dir/ \\
      --mappings path/to/mappings.pkl \\
      --summary  path/to/training_summary.json \\
      [--fp-csv  path/to/FP_AllPairs.csv] \\
      [--model   TransformerModel] \\
      [--top-diseases 10] \\
      [--top-drugs-per-disease 10] \\
      [--max-explanations 50] \\
      [--max-path-length 4] \\
      [--max-paths-per-pair 20] \\
      [--n-bootstrap 1000] \\
      [--explainer-epochs 200]
"""

import os
import json
import math
import pickle
import random
import argparse
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, TransformerConv, GATConv, GINConv, RGCNConv,
)
from torch_geometric.utils import to_networkx
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import (
    ExplanationType, ModelMode, ModelTaskLevel, MaskType,
)


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS  (identical to 2_train_models.py / 3_test_evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers  = num_layers
        self.conv1       = GCNConv(in_channels, hidden_channels)
        self.conv_list   = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
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


class SAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers  = num_layers
        self.conv1       = SAGEConv(in_channels, hidden_channels)
        self.conv_list   = nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
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


class TransformerModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.num_layers  = num_layers
        self.conv1       = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv_list   = nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False)
             for _ in range(num_layers - 1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = self.conv1(x, edge_index)
        x = F.relu(self.ln(x)); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = conv(x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout_rate=0.5, heads=4):
        super().__init__()
        self.num_layers  = num_layers
        self.conv1       = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv_list   = nn.ModuleList(
            [GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
             for _ in range(num_layers - 1)])
        self.ln          = nn.LayerNorm(hidden_channels)
        self.dropout     = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None):
        x = self.conv1(x, edge_index)
        x = F.relu(self.ln(x)); x = self.dropout(x)
        for k, conv in enumerate(self.conv_list):
            x = conv(x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:
                x = F.relu(x); x = self.dropout(x)
        return self.final_layer(x)


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
        for _ in range(num_layers - 1):
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
             for _ in range(num_layers - 1)])
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


# ─────────────────────────────────────────────────────────────────────────────
#  LINK PREDICTOR WRAPPER FOR GNNExplainer
#  ────────────────────────────────────────────────────────────────────────────
#  GNNExplainer needs a model that outputs a scalar (graph-level) for each
#  prediction it explains.  All our GNN backbones output node embeddings, and
#  the actual drug-disease score is a dot-product.  This wrapper makes that
#  single-pair score visible as a "graph-level" output so the Explainer can
#  perturb features/edges and measure how the specific drug→disease score
#  changes — which is the model-faithful explanation we want.
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictorForExplainer(nn.Module):
    """
    Wraps any KG-Bench GNN backbone so that GNNExplainer can produce
    link-prediction explanations.

    Call `set_pair(drug_idx, disease_idx)` before each `explainer(...)` call
    to point the wrapper at the drug-disease pair you want to explain.
    """

    def __init__(self, gnn_model, is_rgcn: bool = False, edge_type=None):
        super().__init__()
        self.gnn       = gnn_model
        self.is_rgcn   = is_rgcn
        self.edge_type = edge_type      # stored for RGCN pass-through
        self.drug_idx     = None
        self.disease_idx  = None

    def set_pair(self, drug_idx: int, disease_idx: int):
        self.drug_idx    = drug_idx
        self.disease_idx = disease_idx

    def forward(self, x, edge_index):
        if self.drug_idx is None or self.disease_idx is None:
            raise RuntimeError("Call set_pair(drug_idx, disease_idx) first.")
        if self.is_rgcn:
            z = self.gnn(x, edge_index, self.edge_type)
        else:
            z = self.gnn(x, edge_index)
        # Dot-product score for the target drug-disease pair → shape [1]
        score = (z[self.drug_idx] * z[self.disease_idx]).sum(dim=-1, keepdim=True)
        return score


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL FROM training_summary.json
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_summary(summary: dict, model_name: str,
                             graph, device) -> nn.Module:
    """
    Reconstruct and load the best trained weights for `model_name` from
    the training_summary produced by 2_train_models.py.
    """
    best = summary["best_models"].get(model_name)
    if best is None:
        raise KeyError(
            f"'{model_name}' not found in training_summary. "
            f"Available: {list(summary['best_models'].keys())}"
        )

    num_layers  = best["num_layers"]
    hidden_dim  = best["hidden_dim"]
    model_path  = best["model_path"]

    kwargs = dict(
        in_channels     = graph.x.size(1),
        hidden_channels = hidden_dim,
        out_channels    = hidden_dim,
        num_layers      = num_layers,
        dropout_rate    = 0.5,
    )
    if model_name == "RGCNModel":
        num_relations = (
            int(graph.metadata.get("num_relations", 6))
            if hasattr(graph, "metadata") else 6
        )
        kwargs["num_relations"] = num_relations

    model_cls = MODEL_CLASSES[model_name]
    model     = model_cls(**kwargs).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Make sure 2_train_models.py has been run and the path is correct."
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  ✓ Loaded {model_name} weights from {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX MAPPINGS  (from mappings.pkl produced by 1_create_graph.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_idx_to_name_type(mappings: dict):
    """
    Build idx_to_name and idx_to_type lookup dicts from the all_mappings pkl.
    Handles gene symbol look-up inside the mappings if available.
    """
    idx_to_name = {}
    idx_to_type = {}

    # Node-type key-mapping names as stored by 1_create_graph.py
    type_info = [
        ("drug_key_mapping",              "Drug"),
        ("drug_type_key_mapping",         "DrugType"),
        ("gene_key_mapping",              "Gene"),
        ("reactome_key_mapping",          "Pathway"),
        ("disease_key_mapping",           "Disease"),
        ("therapeutic_area_key_mapping",  "TherapeuticArea"),
    ]

    # Gene symbol look-up (approvedSymbol stored alongside gene list if present)
    gene_symbol_map = {}
    if "gene_symbol_mapping" in mappings:           # preferred: explicit symbol map
        gene_symbol_map = mappings["gene_symbol_mapping"]

    for mapping_key, node_type in type_info:
        km = mappings.get(mapping_key, {})
        for entity_id, idx in km.items():
            if node_type == "Drug":
                # Use human-readable name list when available
                drug_names = mappings.get("approved_drugs_list_name", [])
                drug_ids   = mappings.get("approved_drugs_list", [])
                if entity_id in drug_ids:
                    pos = drug_ids.index(entity_id)
                    name = drug_names[pos] if pos < len(drug_names) else entity_id
                else:
                    name = entity_id
            elif node_type == "Disease":
                disease_ids   = mappings.get("disease_list", [])
                disease_names = mappings.get("disease_list_name", [])
                if entity_id in disease_ids:
                    pos = disease_ids.index(entity_id)
                    name = disease_names[pos] if pos < len(disease_names) else entity_id
                else:
                    name = entity_id
            elif node_type == "Gene":
                name = gene_symbol_map.get(entity_id, entity_id)
            else:
                name = entity_id

            idx_to_name[int(idx)] = str(name)
            idx_to_type[int(idx)] = node_type

    print(f"  ✓ Built idx mappings: {len(idx_to_name)} nodes")
    return idx_to_name, idx_to_type


# ─────────────────────────────────────────────────────────────────────────────
#  FP PAIR SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def load_fp_pairs_from_csv(fp_csv: str, mappings: dict,
                            top_diseases: int, top_drugs_per_disease: int,
                            sample_size: int = 100_000) -> list:
    """
    Read the FP CSV produced by 3_test_evaluate.py and return a list of
    drug-disease pair dicts with graph node indices attached.

    Strategy: pick the `top_diseases` diseases by average confidence, then
    the `top_drugs_per_disease` highest-confidence drugs for each.
    """
    print(f"  Loading FP candidates from {fp_csv} …")
    df = pd.read_csv(fp_csv, nrows=sample_size)

    # Normalise column names (script 3 uses Drug/Disease/Probability)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("drug", "drug_name"):
            col_map[c] = "drug_name"
        elif lc in ("disease", "disease_name"):
            col_map[c] = "disease_name"
        elif lc in ("probability", "confidence", "score"):
            col_map[c] = "confidence"
    df = df.rename(columns=col_map)

    required = {"drug_name", "disease_name", "confidence"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"FP CSV is missing columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    drug_names    = mappings.get("approved_drugs_list_name", [])
    disease_ids   = mappings.get("disease_list", [])
    disease_names = mappings.get("disease_list_name", [])
    disease_km    = mappings.get("disease_key_mapping", {})

    # Filter to rows where both names are recognisable in the graph
    df = df[df["drug_name"].isin(drug_names) & df["disease_name"].isin(disease_names)]
    print(f"  {len(df):,} FP rows after filtering to known entities")

    # Top diseases by average confidence
    disease_avg = (
        df.groupby("disease_name")["confidence"].mean()
        .sort_values(ascending=False)
        .head(top_diseases)
    )
    top_disease_set = set(disease_avg.index)

    selected_pairs = []
    for disease_name in top_disease_set:
        sub = (df[df["disease_name"] == disease_name]
               .sort_values("confidence", ascending=False)
               .head(top_drugs_per_disease))

        # Locate disease index in the graph
        try:
            d_pos    = disease_names.index(disease_name)
            d_id     = disease_ids[d_pos]
            d_idx    = disease_km[d_id]
        except (ValueError, KeyError):
            continue

        for _, row in sub.iterrows():
            drug_name = row["drug_name"]
            try:
                drug_idx = drug_names.index(drug_name)   # drug offset starts at 0
            except ValueError:
                continue

            selected_pairs.append({
                "drug_name":    drug_name,
                "disease_name": disease_name,
                "drug_idx":     int(drug_idx),
                "disease_idx":  int(d_idx),
                "confidence":   float(row["confidence"]),
            })

    selected_pairs.sort(key=lambda x: x["confidence"], reverse=True)
    print(f"  ✓ Selected {len(selected_pairs)} FP pairs "
          f"({top_diseases} diseases × up to {top_drugs_per_disease} drugs)")
    return selected_pairs


# ─────────────────────────────────────────────────────────────────────────────
#  PATH FINDING
# ─────────────────────────────────────────────────────────────────────────────

def _is_direct_drug_type_first_step(path, drug_idx, idx_to_type):
    """Filter out paths whose first step is Drug → DrugType."""
    if len(path) < 2:
        return True
    if path[0] == drug_idx and idx_to_type.get(path[1], "") == "DrugType":
        return False
    return True


def find_connection_paths(G: nx.Graph, fp_pairs: list,
                          idx_to_name: dict, idx_to_type: dict,
                          max_path_length: int = 4,
                          max_paths_per_pair: int = 20) -> dict:
    """
    For each drug-disease pair, find mechanistic connection paths in the
    knowledge graph using simple-path enumeration (BFS-limited).

    Returns a dict keyed by "drug_name -> disease_name".
    """
    print(f"\nFinding connection paths (max_len={max_path_length}, "
          f"max_paths={max_paths_per_pair}) …")
    all_paths_data = {}

    for pair in tqdm(fp_pairs, desc="Path finding"):
        drug_idx    = pair["drug_idx"]
        disease_idx = pair["disease_idx"]
        pair_key    = f"{pair['drug_name']} -> {pair['disease_name']}"

        try:
            generator      = nx.all_simple_paths(G, drug_idx, disease_idx,
                                                  cutoff=max_path_length)
            valid_paths    = []
            examined       = 0
            exam_limit     = max(500, max_paths_per_pair * 30)

            for path in generator:
                examined += 1
                if _is_direct_drug_type_first_step(path, drug_idx, idx_to_type):
                    valid_paths.append(path)
                    if len(valid_paths) >= max_paths_per_pair:
                        break
                if examined >= exam_limit:
                    break

            if valid_paths:
                valid_paths.sort(key=len)
                path_details = []
                for p in valid_paths:
                    path_details.append({
                        "length": len(p) - 1,
                        "node_indices": p,
                        "nodes": [
                            {"idx":  n,
                             "name": idx_to_name.get(n, f"Node_{n}"),
                             "type": idx_to_type.get(n, "Unknown")}
                            for n in p
                        ],
                    })
                all_paths_data[pair_key] = {
                    "pair":            pair,
                    "paths_found":     len(valid_paths),
                    "paths":           path_details,
                    "shortest_length": min(p["length"] for p in path_details),
                }
            else:
                all_paths_data[pair_key] = {
                    "pair": pair, "paths_found": 0, "paths": [], "shortest_length": 0
                }

        except Exception as exc:
            print(f"  ⚠ Error on {pair_key}: {exc}")
            all_paths_data[pair_key] = {
                "pair": pair, "paths_found": 0, "paths": [], "shortest_length": 0,
                "error": str(exc),
            }

    found = sum(1 for v in all_paths_data.values() if v["paths_found"] > 0)
    print(f"  ✓ {found}/{len(fp_pairs)} pairs have at least one path")
    return all_paths_data


# ─────────────────────────────────────────────────────────────────────────────
#  GNNExplainer – model-faithful explanations
# ─────────────────────────────────────────────────────────────────────────────

class ModelFaithfulExplainer:
    """
    Generates per-pair model-faithful explanations via GNNExplainer.

    Key design decisions aligned with the KG-Bench pipeline:
    - Uses `LinkPredictorForExplainer` wrapper so the explainer optimises
      masks w.r.t. the ACTUAL drug→disease dot-product score, not generic
      node embeddings.
    - Configured as ModelTaskLevel.graph (one scalar output per call) which
      is correct because the wrapper collapses to a single link score.
    - RGCN edge_type is stored in the wrapper and passed through transparently.
    - No global state; all inputs are explicit arguments.
    """

    def __init__(self, model: nn.Module, graph,
                 device, is_rgcn: bool = False,
                 explainer_epochs: int = 200):

        self.graph   = graph
        self.device  = device
        self.is_rgcn = is_rgcn

        # Build the link-predictor wrapper once; update .set_pair() per call
        edge_type = graph.edge_type if (is_rgcn and hasattr(graph, "edge_type")) else None
        self.wrapper = LinkPredictorForExplainer(
            model, is_rgcn=is_rgcn, edge_type=edge_type
        ).to(device)
        self.wrapper.eval()

        # One Explainer instance reused across all pairs (fast)
        self.explainer = Explainer(
            model            = self.wrapper,
            algorithm        = GNNExplainer(epochs=explainer_epochs),
            explanation_type = ExplanationType.model,
            node_mask_type   = MaskType.object,
            edge_mask_type   = MaskType.object,
            model_config     = dict(
                mode        = ModelMode.binary_classification,
                task_level  = ModelTaskLevel.graph,   # wrapper returns a scalar
                return_type = "raw",
            ),
        )

    def explain_pair(self, drug_idx: int, disease_idx: int) -> dict | None:
        """
        Run GNNExplainer for one drug-disease pair.
        Returns a dict with importance scores, or None on failure.
        """
        self.wrapper.set_pair(drug_idx, disease_idx)

        try:
            explanation = self.explainer(
                x          = self.graph.x.float(),
                edge_index = self.graph.edge_index,
            )
        except Exception as exc:
            print(f"    GNNExplainer error: {exc}")
            return None

        return self._process_explanation(explanation, drug_idx, disease_idx)

    def _process_explanation(self, explanation, drug_idx, disease_idx) -> dict | None:
        """Extract and validate importance masks from the explanation object."""

        def _to_1d(t):
            if t is None:
                return torch.empty(0, dtype=torch.float32)
            t = t.float()
            if t.ndim > 1:
                t = t.squeeze(-1) if t.shape[-1] == 1 else t.mean(dim=-1)
            return t

        edge_mask = _to_1d(explanation.get("edge_mask"))
        node_mask = _to_1d(explanation.get("node_mask"))

        if edge_mask.numel() == 0 and node_mask.numel() == 0:
            return None

        # ── Edge importance ────────────────────────────────────────────────
        imp_edge_idx  = torch.tensor([], dtype=torch.long)
        edge_threshold = float("nan")

        if edge_mask.numel() > 0:
            finite  = torch.isfinite(edge_mask)
            valid   = edge_mask[finite]
            if valid.numel() > 0 and valid.max() > valid.min():
                thresh         = torch.quantile(valid, 0.80)
                edge_threshold = float(thresh)
                cmp            = edge_mask.clone()
                cmp[~finite]   = float("-inf")
                imp_edge_idx   = (cmp >= thresh).nonzero(as_tuple=False).squeeze(-1)

        # ── Node importance ────────────────────────────────────────────────
        important_nodes = {drug_idx, disease_idx}
        ei = self.graph.edge_index
        for eidx in imp_edge_idx.tolist():
            important_nodes.add(int(ei[0, eidx]))
            important_nodes.add(int(ei[1, eidx]))

        important_edges = []
        for eidx in imp_edge_idx.tolist():
            imp = float(edge_mask[eidx])
            if math.isfinite(imp):
                important_edges.append({
                    "source":     int(ei[0, eidx]),
                    "target":     int(ei[1, eidx]),
                    "importance": imp,
                    "edge_idx":   int(eidx),
                })

        node_scores = {}
        if node_mask.numel() > 0:
            for nidx in important_nodes:
                if nidx < node_mask.size(0):
                    s = float(node_mask[nidx])
                    if math.isfinite(s):
                        node_scores[int(nidx)] = s

        if not important_edges and not node_scores:
            return None

        return {
            "drug_idx":              drug_idx,
            "disease_idx":           disease_idx,
            "important_nodes":       list(important_nodes),
            "important_edges":       important_edges,
            "num_important_edges":   len(important_edges),
            "node_importance_scores": node_scores,
            "edge_threshold":        edge_threshold,
        }

    def explain_pairs(self, fp_pairs: list, max_explanations: int = 50) -> dict:
        """Run explanations for a list of FP pairs."""
        n   = min(len(fp_pairs), max_explanations)
        out = {}
        print(f"\nGenerating GNNExplainer explanations for {n} pairs …")
        for i, pair in enumerate(tqdm(fp_pairs[:n], desc="GNNExplainer")):
            key = f"{pair['drug_name']} -> {pair['disease_name']}"
            exp = self.explain_pair(pair["drug_idx"], pair["disease_idx"])
            out[key] = {"pair": pair, "explanation": exp,
                        "has_explanation": exp is not None}
            if exp:
                print(f"  ({i+1}/{n}) {key}: "
                      f"{exp['num_important_edges']} imp. edges")
        success = sum(1 for v in out.values() if v["has_explanation"])
        print(f"  ✓ {success}/{n} explanations succeeded")
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  DEGREE-ADJUSTED IMPORTANCE + BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────

def _degree_adjusted_scores(explanations_sample: list, G: nx.Graph,
                             idx_to_type: dict, idx_to_name: dict) -> dict:
    """
    Compute degree-adjusted, globally normalised node importance scores
    for a list of explanation dicts.

    Degree correction: raw_score / log(degree + 2) removes the hub bias
    (high-degree nodes appear important just because they appear everywhere).
    Global normalisation maps the adjusted scores to [0, 1].
    """
    raw_adjusted = []
    entries      = []   # (node_idx, adjusted_score)

    for data in explanations_sample:
        scores = data["explanation"]["node_importance_scores"]
        for nidx, raw in scores.items():
            deg  = G.degree(nidx) if nidx in G else 1
            adj  = raw / math.log(deg + 2)
            raw_adjusted.append(adj)
            entries.append((nidx, adj))

    if not raw_adjusted:
        return {"node_types": {}, "individual_nodes": {}}

    adj_min   = min(raw_adjusted)
    adj_range = max(raw_adjusted) - adj_min
    if adj_range == 0:
        return {"node_types": {}, "individual_nodes": {}}

    type_scores = defaultdict(list)
    name_scores = defaultdict(list)
    for nidx, adj in entries:
        norm  = (adj - adj_min) / adj_range
        ntype = idx_to_type.get(nidx, "Unknown")
        nname = idx_to_name.get(nidx, f"Node_{nidx}")
        type_scores[ntype].append(norm)
        name_scores[nname].append(norm)

    return {
        "node_types": {
            nt: np.mean(ss) for nt, ss in type_scores.items()
            if len(ss) >= 5 and np.std(ss) > 0.01
        },
        "individual_nodes": {
            nn: np.mean(ss) for nn, ss in name_scores.items()
            if len(ss) >= 2 and np.std(ss) > 0.01
        },
    }


def bootstrap_importance(explanations_dict: dict, G: nx.Graph,
                          idx_to_type: dict, idx_to_name: dict,
                          n_bootstrap: int = 1000,
                          confidence: float = 0.95) -> dict:
    """
    Bootstrap confidence intervals for degree-adjusted node importance.
    Only explanations where `has_explanation` is True are included.
    """
    valid = [v for v in explanations_dict.values() if v["has_explanation"]]
    if len(valid) < 5:
        print(f"  ⚠ Only {len(valid)} valid explanations – bootstrap skipped.")
        return {}

    print(f"\nRunning bootstrap analysis ({n_bootstrap} resamples, "
          f"n={len(valid)} explanations) …")

    boot_type  = defaultdict(list)
    boot_name  = defaultdict(list)

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        sample = [valid[j] for j in np.random.randint(0, len(valid), len(valid))]
        stats  = _degree_adjusted_scores(sample, G, idx_to_type, idx_to_name)
        for nt, m in stats["node_types"].items():
            boot_type[nt].append(m)
        for nn, m in stats["individual_nodes"].items():
            boot_name[nn].append(m)

    alpha = 1.0 - confidence
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    def _ci(vals):
        arr = np.array(vals)
        return {
            "mean":       float(np.mean(arr)),
            "ci_lower":   float(np.percentile(arr, lo)),
            "ci_upper":   float(np.percentile(arr, hi)),
            "std":        float(np.std(arr)),
            "n":          len(arr),
            "reliability": "High" if (np.percentile(arr, hi) - np.percentile(arr, lo)) < 0.2 else "Low",
        }

    results = {
        "node_types":       {nt: _ci(vals) for nt, vals in boot_type.items()},
        "individual_nodes": {},
    }

    # Only keep top-20 individual nodes by mean importance
    name_means = {nn: np.mean(vals) for nn, vals in boot_name.items()}
    top20 = sorted(name_means, key=name_means.get, reverse=True)[:20]
    for nn in top20:
        results["individual_nodes"][nn] = _ci(boot_name[nn])

    return results


def print_bootstrap_report(bootstrap_results: dict,
                            idx_to_type: dict, idx_to_name: dict,
                            model_name: str, results_path: str,
                            timestamp: str) -> str:
    """Print and save the bootstrap importance report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"GNNExplainer Attribution Report – {model_name}")
    lines.append(f"Generated: {timestamp}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Degree-adjusted importance corrects for hub bias:")
    lines.append("  adjusted_score = raw_score / log(degree + 2)")
    lines.append("")
    lines.append("Node Type Importance (95% CI):")
    lines.append("-" * 70)
    lines.append(f"  {'Type':<18} {'Mean':>8} {'CI Lower':>10} {'CI Upper':>10}  Reliability")
    lines.append("-" * 70)

    node_types = bootstrap_results.get("node_types", {})
    for nt, s in sorted(node_types.items(), key=lambda x: x[1]["mean"], reverse=True):
        lines.append(
            f"  {nt:<18} {s['mean']:>8.3f} {s['ci_lower']:>10.3f} "
            f"{s['ci_upper']:>10.3f}  {s['reliability']}"
        )

    lines.append("")
    lines.append("Top 20 Individual Nodes (95% CI):")
    lines.append("-" * 70)
    lines.append(
        f"  {'Node':<28} {'Type':<16} {'Mean':>8} {'CI Lower':>10} {'CI Upper':>10}  Rel."
    )
    lines.append("-" * 70)

    ind_nodes = bootstrap_results.get("individual_nodes", {})
    for nn, s in sorted(ind_nodes.items(), key=lambda x: x[1]["mean"], reverse=True):
        nidx = next((k for k, v in idx_to_name.items() if v == nn), None)
        ntype = idx_to_type.get(nidx, "?") if nidx is not None else "?"
        lines.append(
            f"  {nn:<28} {ntype:<16} {s['mean']:>8.3f} {s['ci_lower']:>10.3f} "
            f"{s['ci_upper']:>10.3f}  {s['reliability']}"
        )

    lines.append("")
    lines.append("Reliability guide:")
    lines.append("  High: CI width < 0.20 – trustworthy result")
    lines.append("  Low:  CI width ≥ 0.20 – may reflect noise or insufficient data")

    report = "\n".join(lines)
    print(report)

    path = os.path.join(results_path,
                        f"GNNExplainer_importance_{model_name}_{timestamp}.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"\n  ✓ Saved importance report → {path}")
    return path


def save_importance_csvs(explanations_dict: dict,
                          G: nx.Graph,
                          idx_to_type: dict,
                          idx_to_name: dict,
                          model_name: str,
                          results_path: str,
                          timestamp: str):
    """Export per-node and per-edge importance scores to CSV."""
    node_rows = []
    edge_rows = []

    for pair_key, data in explanations_dict.items():
        if not data["has_explanation"]:
            continue
        exp = data["explanation"]
        drug_name    = data["pair"]["drug_name"]
        disease_name = data["pair"]["disease_name"]
        confidence   = data["pair"]["confidence"]

        for nidx, score in exp["node_importance_scores"].items():
            deg  = G.degree(nidx) if nidx in G else 1
            adj  = score / math.log(deg + 2)
            node_rows.append({
                "drug_name":          drug_name,
                "disease_name":       disease_name,
                "pair_confidence":    confidence,
                "node_idx":           nidx,
                "node_name":          idx_to_name.get(nidx, f"Node_{nidx}"),
                "node_type":          idx_to_type.get(nidx, "Unknown"),
                "node_degree":        deg,
                "raw_importance":     round(score, 6),
                "degree_adj_importance": round(adj, 6),
            })

        for edge in exp["important_edges"]:
            edge_rows.append({
                "drug_name":       drug_name,
                "disease_name":    disease_name,
                "pair_confidence": confidence,
                "src_idx":         edge["source"],
                "src_name":        idx_to_name.get(edge["source"], f"Node_{edge['source']}"),
                "src_type":        idx_to_type.get(edge["source"], "Unknown"),
                "dst_idx":         edge["target"],
                "dst_name":        idx_to_name.get(edge["target"], f"Node_{edge['target']}"),
                "dst_type":        idx_to_type.get(edge["target"], "Unknown"),
                "edge_importance": round(edge["importance"], 6),
            })

    node_csv = os.path.join(results_path,
                            f"GNNExplainer_node_importance_{model_name}_{timestamp}.csv")
    edge_csv = os.path.join(results_path,
                            f"GNNExplainer_edge_importance_{model_name}_{timestamp}.csv")

    pd.DataFrame(node_rows).to_csv(node_csv, index=False)
    pd.DataFrame(edge_rows).to_csv(edge_csv, index=False)
    print(f"  ✓ Node importance CSV → {node_csv}")
    print(f"  ✓ Edge importance CSV → {edge_csv}")
    return node_csv, edge_csv


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK DATA BUILDER  (union of path nodes + explanation nodes)
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPE_COLORS = {
    "Drug":             "#1f77b4",
    "Disease":          "#ff7f0e",
    "Gene":             "#2ca02c",
    "Pathway":          "#17becf",
    "DrugType":         "#9467bd",
    "TherapeuticArea":  "#8c564b",
    "Unknown":          "#999999",
}


def build_network_data(paths_data: dict, explanations_dict: dict,
                        fp_pairs: list,
                        idx_to_name: dict, idx_to_type: dict,
                        G: nx.Graph) -> dict:
    """Collect all nodes and edges visible in the visualizer."""
    all_nodes = set()
    all_edges = set()

    fp_drug_idx    = {p["drug_idx"]    for p in fp_pairs}
    fp_disease_idx = {p["disease_idx"] for p in fp_pairs}

    # Always include FP target nodes
    all_nodes.update(fp_drug_idx)
    all_nodes.update(fp_disease_idx)

    # Path nodes + edges
    for data in paths_data.values():
        for path in data["paths"]:
            nodes = path["node_indices"]
            all_nodes.update(nodes)
            for i in range(len(nodes) - 1):
                all_edges.add(tuple(sorted((nodes[i], nodes[i + 1]))))

    # Explanation nodes + edges
    for data in explanations_dict.values():
        if not data["has_explanation"]:
            continue
        exp = data["explanation"]
        all_nodes.update(exp["important_nodes"])
        for e in exp["important_edges"]:
            all_edges.add(tuple(sorted((e["source"], e["target"]))))

    nodes_data = []
    for nidx in all_nodes:
        ntype  = idx_to_type.get(nidx, "Unknown")
        nname  = idx_to_name.get(nidx, f"Node_{nidx}")
        is_fp  = nidx in fp_drug_idx or nidx in fp_disease_idx
        nodes_data.append({
            "id":           int(nidx),
            "name":         str(nname),
            "type":         str(ntype),
            "color":        NODE_TYPE_COLORS.get(ntype, "#999999"),
            "size":         15,
            "degree":       int(G.degree(nidx)) if nidx in G else 1,
            "is_fp_target": bool(is_fp),
            "is_fp_drug":   bool(nidx in fp_drug_idx),
            "is_fp_disease":bool(nidx in fp_disease_idx),
        })

    edges_data = [{"source": int(s), "target": int(t)} for s, t in all_edges]
    print(f"  ✓ Visualizer network: {len(nodes_data)} nodes, {len(edges_data)} edges")
    return {"nodes": nodes_data, "edges": edges_data}


# ─────────────────────────────────────────────────────────────────────────────
#  HTML VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def create_html_visualization(network_data: dict,
                               paths_data: dict,
                               explanations_dict: dict,
                               fp_pairs: list,
                               idx_to_name: dict,
                               idx_to_type: dict,
                               model_name: str,
                               results_path: str,
                               timestamp: str) -> str:
    """
    Generate an interactive D3.js HTML visualizer with two explanation modes:
      • Path Mode   – mechanistic connectivity paths in the KG
      • Model Mode  – GNNExplainer edge/node importance masks
    """
    # ── Prepare JSON-safe data ────────────────────────────────────────────────
    fp_pairs_info = []
    for data in paths_data.values():
        p    = data["pair"]
        ekey = f"{p['drug_name']} -> {p['disease_name']}"
        edata = explanations_dict.get(ekey, {})
        fp_pairs_info.append({
            "pair_name":      ekey,
            "drug_name":      p["drug_name"],
            "disease_name":   p["disease_name"],
            "confidence":     float(p["confidence"]),
            "drug_idx":       int(p["drug_idx"]),
            "disease_idx":    int(p["disease_idx"]),
            "paths_found":    int(data["paths_found"]),
            "has_explanation":bool(edata.get("has_explanation", False)),
        })
    fp_pairs_info.sort(key=lambda x: x["confidence"], reverse=True)

    # Combine path + model data into one serialisable structure
    combined = {}
    for pair_key, pdata in paths_data.items():
        edata = explanations_dict.get(pair_key, {})
        entry = {
            "pair": {
                "drug_name":    str(pdata["pair"]["drug_name"]),
                "disease_name": str(pdata["pair"]["disease_name"]),
                "confidence":   float(pdata["pair"]["confidence"]),
                "drug_idx":     int(pdata["pair"]["drug_idx"]),
                "disease_idx":  int(pdata["pair"]["disease_idx"]),
            },
            "paths_found":     int(pdata["paths_found"]),
            "paths": [
                {"length": int(p["length"]),
                 "node_indices": [int(n) for n in p["node_indices"]]}
                for p in pdata["paths"]
            ],
            "has_explanation": bool(edata.get("has_explanation", False)),
            "model_explanation": None,
        }
        if edata.get("has_explanation") and edata.get("explanation"):
            exp = edata["explanation"]
            entry["model_explanation"] = {
                "important_nodes": [int(n) for n in exp["important_nodes"]],
                "important_edges": [
                    {"source":     int(e["source"]),
                     "target":     int(e["target"]),
                     "importance": float(e["importance"])}
                    for e in exp["important_edges"]
                ],
                "node_importance_scores": {
                    str(k): float(v)
                    for k, v in exp["node_importance_scores"].items()
                },
            }
        combined[pair_key] = entry

    net_json  = json.dumps(network_data, separators=(",", ":"))
    pair_json = json.dumps(fp_pairs_info, separators=(",", ":"))
    comb_json = json.dumps(combined,      separators=(",", ":"))

    unique_diseases = sorted({p["disease_name"] for p in fp_pairs_info})
    unique_drugs    = sorted({p["drug_name"]    for p in fp_pairs_info})
    disease_opts = "\n".join(
        f'<option value="{d}">{d}</option>' for d in unique_diseases
    )
    drug_opts = "\n".join(
        f'<option value="{d}">{d}</option>' for d in unique_drugs
    )

    paths_success  = sum(1 for p in fp_pairs_info if p["paths_found"] > 0)
    model_success  = sum(1 for p in fp_pairs_info if p["has_explanation"])
    both_success   = sum(1 for p in fp_pairs_info
                         if p["paths_found"] > 0 and p["has_explanation"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KG-Bench GNNExplainer – {model_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#f5f7fa;height:100vh;overflow:hidden;}}
    .container{{display:flex;height:100vh;}}
    .sidebar{{width:380px;background:#fff;border-right:1px solid #ddd;overflow-y:auto;box-shadow:2px 0 8px rgba(0,0,0,.08);display:flex;flex-direction:column;}}
    .header{{padding:16px 20px;background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;flex-shrink:0;}}
    .header h1{{font-size:18px;font-weight:600;}}
    .header p{{font-size:12px;opacity:.85;margin-top:4px;}}
    .mode-bar{{padding:12px 16px;background:#ecf0f1;border-bottom:1px solid #ddd;flex-shrink:0;}}
    .mode-bar h3{{font-size:13px;color:#555;margin-bottom:8px;}}
    .mode-btns{{display:flex;gap:8px;}}
    .mode-btn{{flex:1;padding:7px;border:2px solid #3498db;background:#fff;color:#3498db;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;transition:all .2s;}}
    .mode-btn.active{{background:#3498db;color:#fff;}}
    .mode-hint{{font-size:11px;color:#666;margin-top:6px;line-height:1.4;}}
    .controls{{padding:14px 16px;flex:1;overflow-y:auto;}}
    .ctrl-group{{margin-bottom:16px;}}
    .ctrl-group h3{{font-size:13px;color:#333;margin-bottom:7px;font-weight:600;}}
    select{{width:100%;padding:7px;border:1.5px solid #ccc;border-radius:6px;font-size:13px;background:#fff;}}
    select:focus{{outline:none;border-color:#3498db;}}
    .filter-status{{background:#d6eaf8;border:1px solid #7fb3d3;border-radius:6px;padding:8px;font-size:12px;color:#2c3e50;margin-bottom:10px;display:none;}}
    .clear-btn{{background:#e74c3c;color:#fff;border:none;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;margin-left:6px;}}
    .fp-list{{max-height:380px;overflow-y:auto;}}
    .fp-item{{padding:10px 12px;border:1px solid #eee;border-radius:7px;margin-bottom:7px;cursor:pointer;background:#fff;transition:all .25s;}}
    .fp-item:hover{{background:#f0f7ff;border-color:#3498db;}}
    .fp-item.sel-path{{background:#3498db;color:#fff;border-color:#3498db;}}
    .fp-item.sel-model{{background:#e67e22;color:#fff;border-color:#e67e22;}}
    .fp-title{{font-weight:600;font-size:13px;margin-bottom:3px;}}
    .fp-info{{font-size:11px;opacity:.8;line-height:1.4;}}
    .badge{{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;font-weight:700;margin-left:4px;}}
    .badge-path{{background:#27ae60;color:#fff;}}
    .badge-model{{background:#e67e22;color:#fff;}}
    .badge-none{{background:#bbb;color:#fff;}}
    .stats-box{{padding:12px 16px;background:#f8f9fa;border-top:1px solid #eee;flex-shrink:0;}}
    .stats-box h3{{font-size:13px;color:#333;margin-bottom:8px;font-weight:600;}}
    .stat-row{{display:flex;justify-content:space-between;font-size:12px;color:#666;margin-bottom:4px;}}
    .export-btn{{display:block;width:100%;padding:8px;background:#27ae60;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;margin-bottom:6px;transition:background .2s;}}
    .export-btn:hover{{background:#219a52;}}
    .main-view{{flex:1;position:relative;overflow:hidden;}}
    #net-svg{{width:100%;height:100%;background:#fff;}}
    .link{{stroke:#ccc;stroke-opacity:.7;stroke-width:1.5px;transition:all .3s;}}
    .link.path-hl{{stroke:#3498db;stroke-width:3px;stroke-opacity:1;}}
    .link.model-hl{{stroke:#e67e22;stroke-width:3px;stroke-opacity:1;}}
    .link.dimmed{{opacity:.04;}}
    .node{{cursor:pointer;stroke:#fff;stroke-width:2px;transition:all .3s;}}
    .node.path-hl{{stroke:#2c3e50;stroke-width:3.5px;}}
    .node.model-hl{{stroke:#e67e22;stroke-width:3.5px;}}
    .node.target{{stroke:#000;stroke-width:4.5px;}}
    .node.dimmed{{opacity:.06;}}
    .node-label{{font-size:10px;pointer-events:none;text-anchor:middle;fill:#222;font-weight:600;stroke:#fff;stroke-width:1.2px;paint-order:stroke fill;transition:opacity .3s;}}
    .node-label.dimmed{{opacity:.04;}}
    .legend{{position:absolute;bottom:16px;left:16px;background:rgba(255,255,255,.94);border-radius:10px;padding:12px;box-shadow:0 3px 14px rgba(0,0,0,.1);border:1px solid #eee;}}
    .legend h4{{font-size:12px;color:#333;margin-bottom:8px;font-weight:600;}}
    .legend-row{{display:flex;align-items:center;margin-bottom:5px;font-size:11px;color:#444;}}
    .legend-dot{{width:13px;height:13px;border-radius:50%;margin-right:7px;flex-shrink:0;}}
    .tooltip{{position:absolute;background:rgba(30,30,30,.88);color:#fff;padding:8px 12px;border-radius:7px;font-size:12px;pointer-events:none;display:none;max-width:240px;line-height:1.5;}}
  </style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <div class="header">
      <h1>KG-Bench Explainer</h1>
      <p>Model: {model_name} &nbsp;|&nbsp; {timestamp}</p>
    </div>

    <div class="mode-bar">
      <h3>Explanation Mode</h3>
      <div class="mode-btns">
        <button class="mode-btn active" id="btnPath"  onclick="setMode('path')">🛤️ Path Mode</button>
        <button class="mode-btn"        id="btnModel" onclick="setMode('model')">🧠 Model Mode</button>
      </div>
      <div class="mode-hint">
        <strong>Path:</strong> mechanistic KG connections &nbsp;
        <strong>Model:</strong> GNNExplainer attribution masks
      </div>
    </div>

    <div class="controls">
      <div class="ctrl-group">
        <h3>Filter by Disease</h3>
        <select id="diseaseFilter">
          <option value="">All diseases…</option>
          {disease_opts}
        </select>
      </div>
      <div class="ctrl-group">
        <h3>Filter by Drug</h3>
        <select id="drugFilter">
          <option value="">All drugs…</option>
          {drug_opts}
        </select>
      </div>
      <div class="filter-status" id="filterStatus"></div>

      <div class="ctrl-group">
        <h3>Drug-Disease Pairs
          <span id="pairCount" style="font-weight:400;color:#888;">({len(fp_pairs_info)})</span>
        </h3>
        <div class="fp-list" id="fpList"></div>
      </div>

      <div class="ctrl-group">
        <h3>Export</h3>
        <button class="export-btn" onclick="exportSVG()">📄 Download SVG</button>
        <button class="export-btn" style="background:#2980b9;" onclick="resetHighlight()">↺ Reset View</button>
      </div>
    </div>

    <div class="stats-box">
      <h3>📊 Statistics</h3>
      <div class="stat-row"><span>Pairs:</span><span>{len(fp_pairs_info)}</span></div>
      <div class="stat-row"><span>With paths:</span><span>{paths_success}</span></div>
      <div class="stat-row"><span>With model expl.:</span><span>{model_success}</span></div>
      <div class="stat-row"><span>Both available:</span><span>{both_success}</span></div>
      <div class="stat-row"><span>Nodes:</span><span>{len(network_data['nodes'])}</span></div>
      <div class="stat-row"><span>Edges:</span><span>{len(network_data['edges'])}</span></div>
    </div>
  </div>

  <div class="main-view">
    <svg id="net-svg"></svg>
    <div class="legend">
      <h4>Node Types</h4>
      {''.join(f'<div class="legend-row"><div class="legend-dot" style="background:{c};"></div>{t}</div>' for t, c in NODE_TYPE_COLORS.items() if t != "Unknown")}
    </div>
    <div class="tooltip" id="tooltip"></div>
  </div>
</div>

<script>
const NET  = {net_json};
const PAIRS= {pair_json};
const DATA = {comb_json};
let allPairs    = [...PAIRS];
let filtPairs   = [...PAIRS];
let currentMode = 'path';
let currentSel  = null;

// ── D3 setup ────────────────────────────────────────────────────────────
const svg = d3.select('#net-svg');
const W = () => svg.node().clientWidth;
const H = () => svg.node().clientHeight;

const root = svg.append('g');
const gLinks  = root.append('g');
const gNodes  = root.append('g');
const gLabels = root.append('g');

const sim = d3.forceSimulation(NET.nodes)
  .force('link',      d3.forceLink(NET.edges).id(d=>d.id).distance(80))
  .force('charge',    d3.forceManyBody().strength(-220))
  .force('center',    d3.forceCenter(W()/2, H()/2))
  .force('collision', d3.forceCollide().radius(d=>d.size+8));

const links = gLinks.selectAll('line').data(NET.edges).join('line').attr('class','link');
const nodes = gNodes.selectAll('circle').data(NET.nodes).join('circle')
  .attr('class','node').attr('r',d=>d.size).attr('fill',d=>d.color)
  .call(d3.drag()
    .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on('drag', (e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on('end',  (e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}))
  .on('click', nodeClick)
  .on('mouseover', showTip)
  .on('mouseout',  hideTip);

const labels = gLabels.selectAll('text').data(NET.nodes).join('text')
  .attr('class','node-label')
  .text(d => d.name.length>12 ? d.name.slice(0,12)+'…' : d.name);

sim.on('tick', () => {{
  links.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
       .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  nodes.attr('cx',d=>d.x).attr('cy',d=>d.y);
  labels.attr('x',d=>d.x).attr('y',d=>d.y+22);
}});

svg.call(d3.zoom().scaleExtent([.1,5]).on('zoom', e=>root.attr('transform',e.transform)));

// ── Tooltip ─────────────────────────────────────────────────────────────
const tip = d3.select('#tooltip');
function showTip(e,d){{
  let html = `<strong>${{d.name}}</strong><br>Type: ${{d.type}}<br>Degree: ${{d.degree}}`;
  if(d.is_fp_drug)    html += '<br>⚑ FP Drug';
  if(d.is_fp_disease) html += '<br>⚑ FP Disease';
  tip.style('display','block').style('left',(e.pageX-380+12)+'px')
     .style('top',(e.pageY-16)+'px').html(html);
}}
function hideTip(){{ tip.style('display','none'); }}

// ── Mode switching ───────────────────────────────────────────────────────
function setMode(m){{
  currentMode = m;
  d3.select('#btnPath').classed('active', m==='path');
  d3.select('#btnModel').classed('active', m==='model');
  if(currentSel) highlightPair(currentSel);
}}

// ── Pair highlighting ────────────────────────────────────────────────────
function highlightPair(pairInfo){{
  currentSel = pairInfo;
  d3.selectAll('.fp-item').classed('sel-path',false).classed('sel-model',false);
  d3.select('#fp-'+CSS.escape(pairInfo.pair_name))
    .classed(currentMode==='path'?'sel-path':'sel-model', true);

  const combo = DATA[pairInfo.pair_name];
  if(!combo){{ resetHighlight(); return; }}

  let nodeSet = new Set();
  let edgeSet = new Set();

  if(currentMode === 'path' && combo.paths_found > 0){{
    combo.paths.forEach(p=>{{
      p.node_indices.forEach(n=>nodeSet.add(n));
      for(let i=0;i<p.node_indices.length-1;i++){{
        edgeSet.add([p.node_indices[i],p.node_indices[i+1]].sort().join('-'));
      }}
    }});
  }} else if(currentMode === 'model' && combo.has_explanation && combo.model_explanation){{
    const me = combo.model_explanation;
    me.important_nodes.forEach(n=>nodeSet.add(n));
    me.important_edges.forEach(e=>edgeSet.add([e.source,e.target].sort().join('-')));
  }}

  if(nodeSet.size === 0){{ resetHighlight(); return; }}

  nodeSet.add(pairInfo.drug_idx);
  nodeSet.add(pairInfo.disease_idx);

  nodes.classed('dimmed',    d=>!nodeSet.has(d.id))
       .classed('path-hl',   d=>currentMode==='path'  && nodeSet.has(d.id))
       .classed('model-hl',  d=>currentMode==='model' && nodeSet.has(d.id))
       .classed('target',    d=>d.id===pairInfo.drug_idx||d.id===pairInfo.disease_idx);

  links.classed('dimmed',   function(d){{
         const k=[d.source.id,d.target.id].sort().join('-');
         return !edgeSet.has(k);
       }})
       .classed('path-hl',  function(d){{
         return currentMode==='path'&&edgeSet.has([d.source.id,d.target.id].sort().join('-'));
       }})
       .classed('model-hl', function(d){{
         return currentMode==='model'&&edgeSet.has([d.source.id,d.target.id].sort().join('-'));
       }});

  labels.classed('dimmed', d=>!nodeSet.has(d.id));
}}

function resetHighlight(){{
  nodes.classed('dimmed',false).classed('path-hl',false)
       .classed('model-hl',false).classed('target',false);
  links.classed('dimmed',false).classed('path-hl',false).classed('model-hl',false);
  labels.classed('dimmed',false);
}}

function nodeClick(e,d){{
  if(d.is_fp_disease){{
    applyFilter('disease', d.name);
  }} else if(d.is_fp_drug){{
    applyFilter('drug', d.name);
  }}
}}

// ── Filters ──────────────────────────────────────────────────────────────
function applyFilter(type, val){{
  if(type==='disease'){{
    filtPairs = allPairs.filter(p=>p.disease_name===val);
    d3.select('#diseaseFilter').property('value',val);
    d3.select('#drugFilter').property('value','');
  }} else {{
    filtPairs = allPairs.filter(p=>p.drug_name===val);
    d3.select('#drugFilter').property('value',val);
    d3.select('#diseaseFilter').property('value','');
  }}
  d3.select('#filterStatus').style('display','block')
    .html(`Filtered by ${{type}}: <strong>${{val}}</strong> (${{filtPairs.length}} pairs)
           <button class="clear-btn" onclick="clearFilter()">✕</button>`);
  renderPairList();
}}

function clearFilter(){{
  filtPairs = [...allPairs];
  d3.select('#diseaseFilter').property('value','');
  d3.select('#drugFilter').property('value','');
  d3.select('#filterStatus').style('display','none');
  renderPairList();
}}

d3.select('#diseaseFilter').on('change', function(){{
  if(this.value) applyFilter('disease',this.value); else clearFilter();
}});
d3.select('#drugFilter').on('change', function(){{
  if(this.value) applyFilter('drug',this.value); else clearFilter();
}});

// ── Pair list rendering ──────────────────────────────────────────────────
function renderPairList(){{
  const c = d3.select('#fpList');
  c.selectAll('.fp-item').remove();
  d3.select('#pairCount').text('('+filtPairs.length+')');

  filtPairs.forEach(p=>{{
    const div = c.append('div').attr('class','fp-item')
      .attr('id','fp-'+p.pair_name.replace(/[^a-zA-Z0-9]/g,'_'))
      .on('click',()=>highlightPair(p));
    let badges = '';
    if(p.paths_found>0) badges += '<span class="badge badge-path">PATHS</span>';
    if(p.has_explanation) badges += '<span class="badge badge-model">MODEL</span>';
    if(!p.paths_found&&!p.has_explanation) badges += '<span class="badge badge-none">NONE</span>';
    div.append('div').attr('class','fp-title').html(p.pair_name+badges);
    div.append('div').attr('class','fp-info').html(
      `Confidence: ${{p.confidence.toFixed(4)}}` +
      (p.paths_found>0   ? `<br>📍 ${{p.paths_found}} paths`              : '') +
      (p.has_explanation ? `<br>🧠 Model explanation available`           : '')
    );
  }});
}}

// ── SVG Export ───────────────────────────────────────────────────────────
function exportSVG(){{
  const s = document.getElementById('net-svg');
  const ser = new XMLSerializer();
  const blob = new Blob(['<?xml version="1.0"?>',ser.serializeToString(s)],
                        {{type:'image/svg+xml'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'KGBench_GNNExplainer_{model_name}_{timestamp}.svg';
  a.click();
}}

renderPairList();
console.log('KG-Bench GNNExplainer visualizer loaded – {model_name}');
</script>
</body>
</html>"""

    out_path = os.path.join(results_path,
                            f"GNNExplainer_visualization_{model_name}_{timestamp}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Saved HTML visualizer → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DETECT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_file(directory: str, pattern_parts: tuple) -> str | None:
    """Return the most-recently-modified file in `directory` whose name
    contains ALL strings in `pattern_parts` (case-insensitive)."""
    try:
        files = [
            f for f in os.listdir(directory)
            if all(p.lower() in f.lower() for p in pattern_parts)
        ]
    except FileNotFoundError:
        return None
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)),
               reverse=True)
    return os.path.join(directory, files[0])


def _find_fp_csv(results_path: str, model_name: str) -> str | None:
    """Auto-detect the FP CSV for a given model in results_path."""
    return _find_latest_file(results_path, (model_name, "FP", ".csv"))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_explainer(graph_path: str,
                  results_path: str,
                  mappings_path: str,
                  summary_path: str,
                  fp_csv: str | None,
                  model_name: str,
                  top_diseases: int,
                  top_drugs_per_disease: int,
                  max_explanations: int,
                  max_path_length: int,
                  max_paths_per_pair: int,
                  n_bootstrap: int,
                  explainer_epochs: int,
                  sample_size: int):

    os.makedirs(results_path, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print(f"KG-Bench  4_explainer.py  –  {model_name}")
    print(f"Device : {device}")
    print(f"Outputs: {results_path}")
    print("=" * 70)

    # ── 1. Load graph ────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading graph …  ({graph_path})")
    graph   = torch.load(graph_path, map_location=device)
    graph.x = graph.x.float()
    print(f"  nodes={graph.num_nodes:,}  edges={graph.num_edges:,}  "
          f"features={graph.x.size(1)}")

    # ── 2. Load mappings ─────────────────────────────────────────────────────
    print(f"\n[2/7] Loading mappings … ({mappings_path})")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
    idx_to_name, idx_to_type = build_idx_to_name_type(mappings)

    # ── 3. Load model ────────────────────────────────────────────────────────
    print(f"\n[3/7] Loading trained model … ({model_name})")
    with open(summary_path) as f:
        summary = json.load(f)
    model    = load_model_from_summary(summary, model_name, graph, device)
    is_rgcn  = (model_name == "RGCNModel")

    # ── 4. Load FP pairs ─────────────────────────────────────────────────────
    print(f"\n[4/7] Selecting FP pairs …")
    if fp_csv is None:
        fp_csv = _find_fp_csv(results_path, model_name)
    if fp_csv and os.path.exists(fp_csv):
        fp_pairs = load_fp_pairs_from_csv(fp_csv, mappings,
                                          top_diseases, top_drugs_per_disease,
                                          sample_size)
    else:
        print("  ⚠  No FP CSV found – run 3_test_evaluate.py first, or supply --fp-csv")
        print("  Proceeding with empty pair list; path/explainer steps will be skipped.")
        fp_pairs = []

    if not fp_pairs:
        print("\n  No FP pairs to explain. Exiting.")
        return

    # ── 5. Path finding ──────────────────────────────────────────────────────
    print(f"\n[5/7] Building networkx graph and finding paths …")
    G = to_networkx(graph, to_undirected=True)
    paths_data = find_connection_paths(
        G, fp_pairs, idx_to_name, idx_to_type,
        max_path_length=max_path_length,
        max_paths_per_pair=max_paths_per_pair,
    )

    # ── 6. GNNExplainer ──────────────────────────────────────────────────────
    print(f"\n[6/7] Running GNNExplainer (epochs={explainer_epochs}) …")
    explainer_engine = ModelFaithfulExplainer(
        model, graph, device,
        is_rgcn=is_rgcn,
        explainer_epochs=explainer_epochs,
    )
    explanations_dict = explainer_engine.explain_pairs(fp_pairs, max_explanations)

    # Bootstrap CI + reports
    bootstrap_results = bootstrap_importance(
        explanations_dict, G, idx_to_type, idx_to_name, n_bootstrap=n_bootstrap
    )
    if bootstrap_results:
        print_bootstrap_report(
            bootstrap_results, idx_to_type, idx_to_name,
            model_name, results_path, timestamp
        )

    save_importance_csvs(
        explanations_dict, G, idx_to_type, idx_to_name,
        model_name, results_path, timestamp
    )

    # ── 7. HTML visualizer ───────────────────────────────────────────────────
    print(f"\n[7/7] Building interactive HTML visualizer …")
    network_data = build_network_data(
        paths_data, explanations_dict, fp_pairs,
        idx_to_name, idx_to_type, G
    )
    html_path = create_html_visualization(
        network_data, paths_data, explanations_dict,
        fp_pairs, idx_to_name, idx_to_type,
        model_name, results_path, timestamp
    )

    print("\n" + "=" * 70)
    print("4_explainer.py COMPLETE")
    print("=" * 70)
    print(f"  HTML visualizer → {html_path}")
    print(f"  Importance report (txt) in {results_path}")
    print(f"  Node/edge importance CSVs in {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "KG-Bench 4_explainer.py – GNNExplainer attribution & "
            "interactive visualizer for drug-disease predictions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("graph_path",
                        help="Path to graph .pt file from 1_create_graph.py")
    parser.add_argument("results_path",
                        help="Output directory (also searched for FP CSV)")
    parser.add_argument("--mappings",  default=None,
                        help="Path to all_mappings_*.pkl (auto-detected if omitted)")
    parser.add_argument("--summary",   default=None,
                        help="Path to training_summary_*.json from 2_train_models.py")
    parser.add_argument("--fp-csv",    default=None,
                        help="FP candidates CSV from 3_test_evaluate.py "
                             "(auto-detected in results_path if omitted)")
    parser.add_argument("--model",     default="TransformerModel",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Which trained model to explain")
    parser.add_argument("--top-diseases",         type=int, default=10,
                        help="Number of top diseases by average FP confidence")
    parser.add_argument("--top-drugs-per-disease", type=int, default=10,
                        help="Number of top drugs per disease to explain")
    parser.add_argument("--max-explanations",     type=int, default=50,
                        help="Maximum GNNExplainer calls (subset of selected pairs)")
    parser.add_argument("--max-path-length",      type=int, default=4,
                        help="Maximum hops for networkx path enumeration")
    parser.add_argument("--max-paths-per-pair",   type=int, default=20,
                        help="Maximum paths kept per drug-disease pair")
    parser.add_argument("--n-bootstrap",          type=int, default=1000,
                        help="Bootstrap resampling iterations for CI")
    parser.add_argument("--explainer-epochs",     type=int, default=200,
                        help="GNNExplainer optimisation epochs per pair")
    parser.add_argument("--sample-size",          type=int, default=100_000,
                        help="Max rows to read from FP CSV")

    args = parser.parse_args()

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph not found: {args.graph_path}")

    # Auto-detect mappings
    mappings_path = args.mappings
    if mappings_path is None:
        mappings_path = _find_latest_file(
            "processed_data/mappings/", ("all_mappings",)
        ) or _find_latest_file(args.results_path, ("all_mappings",))
    if mappings_path is None or not os.path.exists(str(mappings_path)):
        raise FileNotFoundError(
            "Could not locate mappings.pkl. Supply --mappings explicitly."
        )

    # Auto-detect training summary
    summary_path = args.summary
    if summary_path is None:
        summary_path = _find_latest_file(
            args.results_path, ("training_summary", ".json")
        )
    if summary_path is None or not os.path.exists(str(summary_path)):
        raise FileNotFoundError(
            "Could not locate training_summary.json. Supply --summary explicitly."
        )

    run_explainer(
        graph_path             = args.graph_path,
        results_path           = args.results_path,
        mappings_path          = str(mappings_path),
        summary_path           = str(summary_path),
        fp_csv                 = args.fp_csv,
        model_name             = args.model,
        top_diseases           = args.top_diseases,
        top_drugs_per_disease  = args.top_drugs_per_disease,
        max_explanations       = args.max_explanations,
        max_path_length        = args.max_path_length,
        max_paths_per_pair     = args.max_paths_per_pair,
        n_bootstrap            = args.n_bootstrap,
        explainer_epochs       = args.explainer_epochs,
        sample_size            = args.sample_size,
    )


if __name__ == "__main__":
    main()
