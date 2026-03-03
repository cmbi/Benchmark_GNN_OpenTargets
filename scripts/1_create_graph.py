#!/usr/bin/env python3
"""
1_create_graph.py  –  KG-Bench: Knowledge Graph Construction
=============================================================
Builds the biomedical knowledge graph from OpenTargets data and produces
all tensors needed by 2_train_models.py and 3_test_evaluate.py.

Supports two modes (auto-detected):
  • processed  – load pre-built CSVs / edge tensors from processed_data/
  • raw        – parse raw OpenTargets parquet files from scratch

Outputs (saved to results_path/):
  {timestamp}_graph.pt            – PyG Data object (all tensors)
  {timestamp}_mappings.pkl        – all node→index mappings + name lists
  training_drug_disease_pairs.csv – human-readable training edges
  processed_data/edges/*.pt       – individual edge tensors (raw mode only)
  processed_data/mappings/*.json  – JSON copies of every mapping dict

Usage:
  python scripts/1_create_graph.py
  python scripts/1_create_graph.py --config config.json --output-dir results/
"""

import os
import json
import pickle
import random
import datetime as dt
import platform
import ast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from pathlib import Path

# ── reproducibility ────────────────────────────────────────────────────────────
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

enable_full_reproducibility(42)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (edit or override via config.json)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_VERSION   = 21.06
VALIDATION_VERSION = 23.06
TEST_VERSION       = 24.06
AS_DATASET         = "associationByOverallDirect"
NEG_SAMPLING       = "random"
PROCESSED_PATH     = "processed_data/"
RESULTS_PATH       = "results/"

def load_config(path: str = "config.json") -> dict:
    """Load config.json if present, otherwise return defaults."""
    defaults = {
        "training_version":    TRAINING_VERSION,
        "validation_version":  VALIDATION_VERSION,
        "test_version":        TEST_VERSION,
        "as_dataset":          AS_DATASET,
        "negative_sampling_approach": NEG_SAMPLING,
        "processed_path":      PROCESSED_PATH,
        "results_path":        RESULTS_PATH,
    }
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults

CFG = load_config()

TRAINING_VERSION   = float(CFG["training_version"])
VALIDATION_VERSION = float(CFG["validation_version"])
TEST_VERSION       = float(CFG["test_version"])
AS_DATASET         = CFG["as_dataset"]
NEG_SAMPLING       = CFG["negative_sampling_approach"]
PROCESSED_PATH     = CFG["processed_path"]
RESULTS_PATH       = CFG.get("results_path", RESULTS_PATH)

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(f"{PROCESSED_PATH}mappings/", exist_ok=True)
os.makedirs(f"{PROCESSED_PATH}tables/", exist_ok=True)
os.makedirs(f"{PROCESSED_PATH}edges/", exist_ok=True)

# ── raw data paths (platform-aware) ───────────────────────────────────────────
if platform.system() == "Windows":
    GENERAL_PATH = r"C:\\OpenTargets_datasets\\downloads\\"
else:
    GENERAL_PATH = "OT/"

def raw_path(version, dataset):
    if platform.system() == "Windows":
        return f"{GENERAL_PATH}{version}\\{dataset}"
    return f"{GENERAL_PATH}{version}/{dataset}"

INDICATION_PATH     = raw_path(TRAINING_VERSION,   "indication")
VAL_INDICATION_PATH = raw_path(VALIDATION_VERSION, "indication")
TEST_INDICATION_PATH= raw_path(TEST_VERSION,        "indication")
MOLECULE_PATH       = raw_path(TRAINING_VERSION,   "molecule")
DISEASE_PATH        = raw_path(TRAINING_VERSION,   "diseases")
GENE_PATH           = raw_path(TRAINING_VERSION,   "targets")
ASSOC_PATH          = raw_path(TRAINING_VERSION,   AS_DATASET)

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS  (mirrors main_script)
# ─────────────────────────────────────────────────────────────────────────────

def extract_edges(table, src_mapping, dst_mapping,
                  return_edge_list=False, return_edge_set=False):
    src_col  = table.column(0).combine_chunks()
    dst_col  = table.column(1).combine_chunks()
    edges = []
    for i in range(len(src_col)):
        s = src_col[i].as_py()
        targets = dst_col.slice(i, 1).to_pylist()[0]
        if not isinstance(targets, list):
            targets = [targets]
        for t in targets:
            if s in src_mapping and t in dst_mapping:
                edges.append((src_mapping[s], dst_mapping[t]))
    if return_edge_list:
        return edges
    if return_edge_set:
        return set(edges)
    unique_edges = list(set(edges))
    return torch.tensor(unique_edges, dtype=torch.long).t().contiguous()


def extract_test_edges(table, src_mapping, dst_mapping):
    src_col = table.column(0).combine_chunks()
    dst_col = table.column(1).combine_chunks()
    edges = []
    for i in range(len(src_col)):
        s = src_col[i].as_py()
        targets = dst_col.slice(i, 1).to_pylist()[0]
        if not isinstance(targets, list):
            targets = [targets]
        for t in targets:
            if s in src_mapping and t in dst_mapping:
                edges.append((src_mapping[s], dst_mapping[t]))
    return set(edges)


def boolean_encode(array, pad_length):
    s = pd.Series(array.to_pandas()).astype("float")
    arr = s.fillna(-1).to_numpy().reshape(-1, 1)
    tensor = torch.from_numpy(arr.astype(np.int64)).float()
    pad = len(pad_length) - tensor.shape[0]
    if pad > 0:
        tensor = F.pad(tensor, (0, 0, 0, pad), value=-1)
    return tensor


def normalize(array, pad_length):
    df = pd.DataFrame(array.to_pandas().to_numpy().reshape(-1, 1))
    df.fillna(-1, inplace=True)
    std = df.std().item()
    standardized = (df - df.mean()) / std if std != 0 else df - df.mean()
    tensor = torch.from_numpy(standardized.to_numpy()).float()
    pad = len(pad_length) - tensor.shape[0]
    if pad > 0:
        tensor = F.pad(tensor, (0, 0, 0, pad), value=-1)
    return tensor


def cat_encode(array, pad_length):
    uni = array.unique().to_pandas()
    unidict = {uni[i]: i for i in range(len(uni))}
    tensor = torch.tensor([unidict[v] for v in array.to_pandas()],
                          dtype=torch.float32).unsqueeze(1)
    pad = len(pad_length) - tensor.shape[0]
    if pad > 0:
        tensor = F.pad(tensor, (0, 0, 0, pad), value=-1)
    return tensor


def pad_feature_matrix(matrix, pad_size, pad_value=-1):
    if matrix.size(1) < pad_size:
        padding = torch.ones(matrix.size(0), pad_size - matrix.size(1)) * pad_value
        matrix = torch.cat([matrix, padding], dim=1)
    return matrix


# Global feature column order  (matches main_script exactly)
GLOBAL_FEATURE_COLUMNS = [
    "drug_one_hot",           # 0
    "drug_type_one_hot",      # 1
    "gene_one_hot",           # 2
    "disease_one_hot",        # 3
    "reactome_one_hot",       # 4
    "therapeutic_area_one_hot",# 5
    "blackBoxWarning",        # 6
    "yearOfFirstApproval",    # 7
    "bioType",                # 8
]
PAD_SIZE = len(GLOBAL_FEATURE_COLUMNS)  # 9


def align_features(matrix, feature_columns, global_feature_columns=GLOBAL_FEATURE_COLUMNS):
    """Place each column at its global position; fill unused positions with -1."""
    aligned = torch.zeros(matrix.size(0), len(global_feature_columns)) - 1
    for idx, col in enumerate(feature_columns):
        g_idx = global_feature_columns.index(col)
        aligned[:, g_idx] = matrix[:, idx]
    return aligned


# One-hot node type vectors  (order: drug, drug_type, gene, disease, reactome, TA)
NODE_TYPE_ONEHOT = {
    "drug":             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "drug_type":        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "gene":             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "disease":          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "reactome":         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "therapeutic_area": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

# ─────────────────────────────────────────────────────────────────────────────
#  ID REDUNDANCY MAPPINGS  (identical to main_script)
# ─────────────────────────────────────────────────────────────────────────────
REDUNDANT_DRUG_MAPPING = {
    'CHEMBL1200538': 'CHEMBL632',  'CHEMBL1200376': 'CHEMBL632',
    'CHEMBL1200384': 'CHEMBL632',  'CHEMBL1201207': 'CHEMBL632',
    'CHEMBL1497':    'CHEMBL632',  'CHEMBL1201661': 'CHEMBL3989767',
    'CHEMBL1506':    'CHEMBL130',  'CHEMBL1201281': 'CHEMBL130',
    'CHEMBL1201289': 'CHEMBL1753', 'CHEMBL3184512': 'CHEMBL1753',
    'CHEMBL1530428': 'CHEMBL384467','CHEMBL1201302': 'CHEMBL384467',
    'CHEMBL1511':    'CHEMBL135',  'CHEMBL4298187': 'CHEMBL2108597',
    'CHEMBL4298110': 'CHEMBL2108597','CHEMBL1200640':'CHEMBL2108597',
    'CHEMBL989':     'CHEMBL1501', 'CHEMBL1201064': 'CHEMBL1200600',
    'CHEMBL1473':    'CHEMBL1676', 'CHEMBL1201512': 'CHEMBL1201688',
    'CHEMBL1201657': 'CHEMBL1201513','CHEMBL1091':  'CHEMBL389621',
    'CHEMBL1549':    'CHEMBL389621','CHEMBL3989663':'CHEMBL389621',
    'CHEMBL1641':    'CHEMBL389621','CHEMBL1200562':'CHEMBL389621',
    'CHEMBL1201544': 'CHEMBL2108597','CHEMBL1200823':'CHEMBL2108597',
    'CHEMBL2021423': 'CHEMBL1200572','CHEMBL1364144':'CHEMBL650',
    'CHEMBL1200844': 'CHEMBL650',  'CHEMBL1201265': 'CHEMBL650',
    'CHEMBL1140':    'CHEMBL573',  'CHEMBL1152':    'CHEMBL131',
    'CHEMBL1201231': 'CHEMBL131',  'CHEMBL1200909': 'CHEMBL131',
    'CHEMBL635':     'CHEMBL131',  'CHEMBL1200335': 'CHEMBL386630',
    'CHEMBL1504':    'CHEMBL1451', 'CHEMBL1200449': 'CHEMBL1451',
    'CHEMBL1200878': 'CHEMBL1451', 'CHEMBL1200929': 'CHEMBL3988900',
}
REDUNDANT_DISEASE_MAPPING = {
    'EFO_1000905': 'EFO_0004228',
    'EFO_0005752': 'EFO_1001888',
    'EFO_0007512': 'EFO_0007510',
}


def resolve_mapping(chembl_id, mapping_dict):
    visited = set()
    while chembl_id in mapping_dict and chembl_id not in visited:
        visited.add(chembl_id)
        chembl_id = mapping_dict[chembl_id]
    return chembl_id


def safe_list_conversion(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def update_approved_indications(disease_list, mapping_dict):
    if not isinstance(disease_list, list):
        return disease_list
    return [mapping_dict.get(str(d), str(d)) for d in disease_list]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def detect_data_mode():
    """Return 'processed' if all pre-built CSVs exist, else 'raw'."""
    required = [
        f"{PROCESSED_PATH}tables/processed_molecules.csv",
        f"{PROCESSED_PATH}tables/processed_indications.csv",
        f"{PROCESSED_PATH}tables/processed_diseases.csv",
        f"{PROCESSED_PATH}edges/1_molecule_drugType_edges.pt",
    ]
    return "processed" if all(os.path.exists(p) for p in required) else "raw"


def load_raw_data():
    """Load and filter raw OpenTargets parquet files (mirrors main_script)."""
    print("  [raw] Loading indication parquet …")
    indication_dataset = ds.dataset(INDICATION_PATH, format="parquet")
    indication_table   = indication_dataset.to_table()

    expr = pc.list_value_length(pc.field("approvedIndications")) > 0
    filtered_indication_table = indication_table.filter(expr)
    approvedDrugs = filtered_indication_table.column("id").combine_chunks()

    print("  [raw] Loading molecule parquet …")
    molecule_dataset = ds.dataset(MOLECULE_PATH, format="parquet")
    molecule_table   = molecule_dataset.to_table()

    # Normalise drugType column
    drug_type_col = pc.replace_substring(molecule_table.column("drugType"), "unknown", "Unknown")
    fill_val      = pa.scalar("Unknown", type=pa.string())
    molecule_table = (molecule_table.drop_columns("drugType")
                                    .add_column(3, "drugType", drug_type_col.fill_null(fill_val)))

    # Flatten nested columns we actually need
    filtered_molecule_table = molecule_table.select(
        ["id","name","drugType","blackBoxWarning","yearOfFirstApproval",
         "parentId","childChemblIds","linkedDiseases","hasBeenWithdrawn","linkedTargets"]
    ).flatten().drop_columns(["linkedTargets.count","linkedDiseases.count"])

    filtered_molecule_df = filtered_molecule_table.to_pandas()

    # Remove child molecules (keep parent representatives only)
    filtered_molecule_df = filtered_molecule_df[pd.isna(filtered_molecule_df["parentId"])]

    # ── Apply drug ID redundancy remapping ────────────────────────────────────
    id_to_parent = {k: resolve_mapping(v, REDUNDANT_DRUG_MAPPING)
                    for k, v in REDUNDANT_DRUG_MAPPING.items()}

    filtered_indication_df = filtered_indication_table.to_pandas()
    filtered_indication_df["id"] = filtered_indication_df["id"].apply(
        lambda x: resolve_mapping(x, id_to_parent) if x in id_to_parent else x
    )
    filtered_indication_df["approvedIndications"] = (
        filtered_indication_df["approvedIndications"].apply(safe_list_conversion)
    )

    # Apply disease ID remapping inside approvedIndications lists
    id_to_disease = {k: resolve_mapping(v, REDUNDANT_DISEASE_MAPPING)
                     for k, v in REDUNDANT_DISEASE_MAPPING.items()}
    filtered_indication_df["approvedIndications"] = (
        filtered_indication_df["approvedIndications"].apply(
            lambda x: update_approved_indications(x, id_to_disease)
        )
    )

    # Keep only drugs that have valid indication entries after mapping
    unique_chembl_ids = filtered_indication_df["id"].unique()
    filtered_molecule_df = filtered_molecule_df[
        filtered_molecule_df["id"].isin(unique_chembl_ids)
    ]

    print("  [raw] Loading disease parquet …")
    disease_dataset = ds.dataset(DISEASE_PATH, format="parquet")
    disease_table   = disease_dataset.to_table()

    # Filter out phenotypic/measurement-only diseases (EFO_0001444)
    disease_table = disease_table.filter(
        pc.list_value_length(pc.field("therapeuticAreas")) > 0
    )
    df_dis = disease_table.to_pandas()
    df_dis = df_dis[~df_dis["therapeuticAreas"].apply(lambda x: "EFO_0001444" in x)]
    disease_table = pa.Table.from_pandas(df_dis)

    disease_table = disease_table.select(
        ["id","name","description","ancestors","descendants","children","therapeuticAreas"]
    )

    # Filter out non-disease ontology prefixes
    prefixes = ["UBERON","ZFA","CL","GO","FBbt","FMA"]
    cond = prefixes[0]
    combined = pc.starts_with(disease_table.column("id"), cond)
    for p in prefixes[1:]:
        combined = pc.or_(combined, pc.starts_with(disease_table.column("id"), p))
    disease_table = disease_table.filter(pc.invert(combined))

    # Keep only leaf diseases (no descendants), remove specific IDs
    disease_table = disease_table.filter(
        pc.list_value_length(pc.field("descendants")) == 0
    )
    for remove_id in ["EFO_0000544","EFO_1000905","EFO_0005752","EFO_0007512"]:
        disease_table = disease_table.filter(pc.field("id") != remove_id)

    # Apply disease ID remapping to the disease table itself
    df_dis2 = disease_table.to_pandas()
    filtered_disease_df = df_dis2.copy()

    print("  [raw] Loading gene/target parquet …")
    gene_dataset = ds.dataset(GENE_PATH, format="parquet")
    gene_table   = gene_dataset.to_table().flatten().flatten()

    # Filter genes linked to approved drugs
    linked_genes = (
        pa.Table.from_pandas(filtered_molecule_df)
        .select(["id","linkedTargets.rows"]).drop_null()
        .column("linkedTargets.rows").combine_chunks()
    )
    gene_filter = pc.is_in(gene_table.column("id"),
                           value_set=pc.unique(linked_genes.flatten()))
    filtered_gene_table = gene_table.filter(gene_filter)

    if TRAINING_VERSION in (21.04, 21.06):
        filtered_gene_table = filtered_gene_table.select(
            ["id","approvedName","bioType","proteinAnnotations.functions","reactome"]
        ).flatten()
        gene_reactome_table = filtered_gene_table.select(["id","reactome"]).flatten()
    else:
        filtered_gene_table = filtered_gene_table.select(
            ["id","approvedName","biotype","functionDescriptions","pathways"]
        ).flatten()
        gene_rt_df = filtered_gene_table.select(["id","pathways"]).to_pandas()
        exploded   = gene_rt_df.explode("pathways")
        exploded["pathwayId"] = exploded["pathways"].apply(
            lambda x: x["pathwayId"] if pd.notnull(x) else None
        )
        gene_reactome_table = pa.Table.from_pandas(
            exploded[["id","pathwayId"]].dropna()
        )

    print("  [raw] Loading associations parquet …")
    assoc_dataset = ds.dataset(ASSOC_PATH, format="parquet")
    assoc_table   = assoc_dataset.to_table()

    score_col = next(
        (c for c in assoc_table.column_names if "score" in c.lower() or "Score" in c),
        None
    )
    assoc_table = assoc_table.select(["diseaseId","targetId",score_col])

    gene_filter2  = pc.is_in(assoc_table.column("targetId"),
                              value_set=pc.unique(linked_genes.flatten()))
    gene_filt_assoc = assoc_table.filter(gene_filter2)

    disease_ids = pa.array(filtered_disease_df["id"].unique().tolist())
    dis_filter  = pc.is_in(gene_filt_assoc.column("diseaseId"), value_set=disease_ids)
    filtered_assoc_table = gene_filt_assoc.filter(dis_filter)
    filtered_assoc_table = filtered_assoc_table.filter(
        pc.field(score_col) >= 0.01
    )

    return (
        filtered_molecule_df, filtered_indication_df,
        filtered_disease_df, filtered_gene_table,
        gene_reactome_table, filtered_assoc_table,
    )


def load_processed_data():
    """Load pre-built CSV tables and edge tensors."""
    print("  [processed] Loading pre-processed CSVs …")
    td = f"{PROCESSED_PATH}tables/"
    filtered_molecule_df    = pd.read_csv(f"{td}processed_molecules.csv")
    filtered_indication_df  = pd.read_csv(f"{td}processed_indications.csv")
    filtered_disease_df     = pd.read_csv(f"{td}processed_diseases.csv")
    filtered_gene_df        = pd.read_csv(f"{td}processed_genes.csv")
    filtered_assoc_df       = pd.read_csv(f"{td}processed_associations.csv")

    # Restore list column
    if "approvedIndications" in filtered_indication_df.columns:
        filtered_indication_df["approvedIndications"] = (
            filtered_indication_df["approvedIndications"].apply(safe_list_conversion)
        )

    gene_table = pa.Table.from_pandas(filtered_gene_df)

    # Reactome table depends on version
    if TRAINING_VERSION in (21.04, 21.06):
        gene_table_full = gene_table
        gene_reactome_table = (
            gene_table_full.select(["id","reactome"]).flatten()
            if "reactome" in gene_table_full.column_names
            else pa.Table.from_pandas(pd.DataFrame({"id":[],"pathwayId":[]}))
        )
    else:
        if "pathwayId" in filtered_gene_df.columns:
            gene_reactome_table = pa.Table.from_pandas(
                filtered_gene_df[["id","pathwayId"]].dropna()
            )
        else:
            gene_reactome_table = pa.Table.from_pandas(pd.DataFrame({"id":[],"pathwayId":[]}))

    assoc_table = pa.Table.from_pandas(filtered_assoc_df)
    return (
        filtered_molecule_df, filtered_indication_df,
        filtered_disease_df, gene_table,
        gene_reactome_table, assoc_table,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  NODE MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

def build_node_mappings(filtered_molecule_df, filtered_disease_df,
                        filtered_gene_table, gene_reactome_table,
                        filtered_molecule_table_pa):
    """Build all node→index dicts (identical offset scheme to main_script)."""

    approved_drugs_list      = filtered_molecule_df["id"].tolist()
    approved_drugs_list_name = filtered_molecule_df["name"].tolist()

    drug_type_col  = filtered_molecule_table_pa.column("drugType").combine_chunks()
    drug_type_list = drug_type_col.drop_null().unique().to_pylist()

    gene_col  = filtered_gene_table.column("id").combine_chunks()
    gene_list = gene_col.unique().to_pylist()

    # Reactome column differs by version
    if TRAINING_VERSION in (21.04, 21.06):
        reactome_raw = filtered_gene_table.column("reactome").combine_chunks().flatten()
    elif "pathwayId" in gene_reactome_table.column_names:
        reactome_raw = gene_reactome_table.column("pathwayId").combine_chunks()
    else:
        reactome_raw = pa.array([], type=pa.string())
    reactome_list = reactome_raw.unique().to_pylist()

    disease_list      = filtered_disease_df["id"].tolist()
    disease_list_name = filtered_disease_df["name"].tolist()

    ta_col   = filtered_disease_df["therapeuticAreas"].apply(safe_list_conversion)
    ta_flat  = [item for sublist in ta_col for item in sublist]
    therapeutic_area_list = list(dict.fromkeys(ta_flat))  # unique, order-preserving

    # ── Build offset-chained mappings ─────────────────────────────────────────
    o0 = 0
    drug_key_mapping = {d: i + o0 for i, d in enumerate(approved_drugs_list)}

    o1 = len(drug_key_mapping)
    drug_type_key_mapping = {t: i + o1 for i, t in enumerate(drug_type_list)}

    o2 = o1 + len(drug_type_key_mapping)
    gene_key_mapping = {g: i + o2 for i, g in enumerate(gene_list)}

    o3 = o2 + len(gene_key_mapping)
    reactome_key_mapping = {r: i + o3 for i, r in enumerate(reactome_list)}

    o4 = o3 + len(reactome_key_mapping)
    disease_key_mapping = {d: i + o4 for i, d in enumerate(disease_list)}
    disease_offset = o4   # saved in metadata for downstream scripts

    o5 = o4 + len(disease_key_mapping)
    therapeutic_area_key_mapping = {t: i + o5 for i, t in enumerate(therapeutic_area_list)}

    mappings = dict(
        approved_drugs_list=approved_drugs_list,
        approved_drugs_list_name=approved_drugs_list_name,
        drug_type_list=drug_type_list,
        gene_list=gene_list,
        reactome_list=reactome_list,
        disease_list=disease_list,
        disease_list_name=disease_list_name,
        therapeutic_area_list=therapeutic_area_list,
        drug_key_mapping=drug_key_mapping,
        drug_type_key_mapping=drug_type_key_mapping,
        gene_key_mapping=gene_key_mapping,
        reactome_key_mapping=reactome_key_mapping,
        disease_key_mapping=disease_key_mapping,
        therapeutic_area_key_mapping=therapeutic_area_key_mapping,
        disease_offset=disease_offset,
    )
    return mappings


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE CONSTRUCTION  (9-dim, aligned with main_script)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(filtered_molecule_table_pa, filtered_gene_table, mappings):
    m = mappings
    drug_indices = torch.tensor([m["drug_key_mapping"][d]
                                  for d in m["approved_drugs_list"]], dtype=torch.long)
    drug_type_indices = torch.tensor([m["drug_type_key_mapping"][t]
                                       for t in m["drug_type_list"]], dtype=torch.long)
    gene_indices = torch.tensor([m["gene_key_mapping"][g]
                                  for g in m["gene_list"]], dtype=torch.long)
    reactome_indices = torch.tensor([m["reactome_key_mapping"][r]
                                      for r in m["reactome_list"]], dtype=torch.long)
    disease_indices = torch.tensor([m["disease_key_mapping"][d]
                                     for d in m["disease_list"]], dtype=torch.long)
    ta_indices = torch.tensor([m["therapeutic_area_key_mapping"][t]
                                 for t in m["therapeutic_area_list"]], dtype=torch.long)

    # ── Drug features ──────────────────────────────────────────────────────────
    bbw  = filtered_molecule_table_pa.column("blackBoxWarning").combine_chunks()
    year = filtered_molecule_table_pa.column("yearOfFirstApproval").combine_chunks()
    bbw_vec  = boolean_encode(bbw, drug_indices)
    year_vec = normalize(year, drug_indices)

    drug_oh  = torch.tensor([NODE_TYPE_ONEHOT["drug"]], dtype=torch.float32).repeat(len(drug_indices), 1)
    drug_raw = torch.cat([drug_oh, bbw_vec, year_vec], dim=1)          # [N_drug, 8]
    drug_raw = pad_feature_matrix(drug_raw, PAD_SIZE)                  # [N_drug, 9]

    drug_feature_columns = [
        "drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
        "reactome_one_hot","therapeutic_area_one_hot","blackBoxWarning","yearOfFirstApproval",
    ]
    aligned_drug = align_features(drug_raw, drug_feature_columns)

    # ── Drug-type features ─────────────────────────────────────────────────────
    dt_oh  = torch.tensor([NODE_TYPE_ONEHOT["drug_type"]], dtype=torch.float32).repeat(len(drug_type_indices), 1)
    dt_raw = pad_feature_matrix(dt_oh, PAD_SIZE)
    dt_feature_columns = [
        "drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
        "reactome_one_hot","therapeutic_area_one_hot",
    ]
    aligned_drug_type = align_features(dt_raw, dt_feature_columns)

    # ── Gene features ──────────────────────────────────────────────────────────
    biotype_col = (
        filtered_gene_table.column("bioType") if "bioType" in filtered_gene_table.column_names
        else filtered_gene_table.column("biotype")
    ).combine_chunks()
    biotype_vec = cat_encode(biotype_col, gene_indices)                # [N_gene, 1]
    gene_oh     = torch.tensor([NODE_TYPE_ONEHOT["gene"]], dtype=torch.float32).repeat(len(gene_indices), 1)
    gene_raw    = torch.cat([gene_oh, biotype_vec], dim=1)             # [N_gene, 7]
    gene_raw    = pad_feature_matrix(gene_raw, PAD_SIZE)               # [N_gene, 9]

    gene_feature_columns = [
        "drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
        "reactome_one_hot","therapeutic_area_one_hot","bioType",
    ]
    aligned_gene = align_features(gene_raw, gene_feature_columns)

    # ── Reactome features ──────────────────────────────────────────────────────
    r_oh  = torch.tensor([NODE_TYPE_ONEHOT["reactome"]], dtype=torch.float32).repeat(len(reactome_indices), 1)
    r_raw = pad_feature_matrix(r_oh, PAD_SIZE)
    r_fc  = ["drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
             "reactome_one_hot","therapeutic_area_one_hot"]
    aligned_reactome = align_features(r_raw, r_fc)

    # ── Disease features ───────────────────────────────────────────────────────
    dis_oh  = torch.tensor([NODE_TYPE_ONEHOT["disease"]], dtype=torch.float32).repeat(len(disease_indices), 1)
    dis_raw = pad_feature_matrix(dis_oh, PAD_SIZE)
    dis_fc  = ["drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
               "reactome_one_hot","therapeutic_area_one_hot"]
    aligned_disease = align_features(dis_raw, dis_fc)

    # ── Therapeutic-area features ──────────────────────────────────────────────
    ta_oh  = torch.tensor([NODE_TYPE_ONEHOT["therapeutic_area"]], dtype=torch.float32).repeat(len(ta_indices), 1)
    ta_raw = pad_feature_matrix(ta_oh, PAD_SIZE)
    ta_fc  = ["drug_one_hot","drug_type_one_hot","gene_one_hot","disease_one_hot",
              "reactome_one_hot","therapeutic_area_one_hot"]
    aligned_ta = align_features(ta_raw, ta_fc)

    all_features = torch.vstack([
        aligned_drug, aligned_drug_type, aligned_gene,
        aligned_disease, aligned_reactome, aligned_ta
    ]).float()

    print(f"Feature matrix shape: {all_features.shape}")
    return all_features


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_edges(filtered_molecule_table_pa, filtered_indication_table_pa,
                filtered_disease_table_pa, gene_reactome_table,
                filtered_assoc_table, mappings):
    m = mappings

    mol_dt_table   = filtered_molecule_table_pa.select(["id","drugType"]).drop_null().flatten()
    mol_dis_table  = filtered_indication_table_pa.select(["id","approvedIndications"]).flatten()
    mol_gene_table = filtered_molecule_table_pa.select(["id","linkedTargets.rows"]).drop_null().flatten()
    dis_ta_table   = filtered_disease_table_pa.select(["id","therapeuticAreas"]).drop_null().flatten()
    dis_gene_table = filtered_assoc_table.select(["diseaseId","targetId"]).flatten()

    print("  Extracting edges …")
    mol_dt_edges   = extract_edges(mol_dt_table,   m["drug_key_mapping"], m["drug_type_key_mapping"])
    mol_dis_edges  = extract_edges(mol_dis_table,  m["drug_key_mapping"], m["disease_key_mapping"])
    mol_gene_edges = extract_edges(mol_gene_table, m["drug_key_mapping"], m["gene_key_mapping"])
    gene_rt_edges  = extract_edges(gene_reactome_table, m["gene_key_mapping"], m["reactome_key_mapping"])
    dis_ta_edges   = extract_edges(dis_ta_table,   m["disease_key_mapping"], m["therapeutic_area_key_mapping"])
    dis_gene_edges = extract_edges(dis_gene_table, m["disease_key_mapping"], m["gene_key_mapping"])

    # De-duplicate
    for name, tensor in [("mol_dt", mol_dt_edges), ("mol_dis", mol_dis_edges),
                         ("mol_gene", mol_gene_edges), ("gene_rt", gene_rt_edges),
                         ("dis_ta", dis_ta_edges), ("dis_gene", dis_gene_edges)]:
        if tensor.numel() > 0:
            tensor = torch.unique(tensor, dim=1)

    all_edge_index = torch.cat([mol_dt_edges, mol_dis_edges, mol_gene_edges,
                                gene_rt_edges, dis_ta_edges, dis_gene_edges], dim=1)

    # ── Edge type tensor for RGCN (6 relation types, 0–5) ─────────────────────
    edge_types = torch.cat([
        torch.zeros(mol_dt_edges.size(1),   dtype=torch.long),   # 0 Drug→DrugType
        torch.ones(mol_dis_edges.size(1),   dtype=torch.long),   # 1 Drug→Disease
        torch.full((mol_gene_edges.size(1),), 2, dtype=torch.long),  # 2 Drug→Gene
        torch.full((gene_rt_edges.size(1),),  3, dtype=torch.long),  # 3 Gene→Reactome
        torch.full((dis_ta_edges.size(1),),   4, dtype=torch.long),  # 4 Disease→TherArea
        torch.full((dis_gene_edges.size(1),), 5, dtype=torch.long),  # 5 Disease→Gene
    ])

    edges = dict(
        molecule_drugType_edges=mol_dt_edges,
        molecule_disease_edges=mol_dis_edges,
        molecule_gene_edges=mol_gene_edges,
        gene_reactome_edges=gene_rt_edges,
        disease_therapeutic_edges=dis_ta_edges,
        disease_gene_edges=dis_gene_edges,
        all_edge_index=all_edge_index,
        edge_types=edge_types,
    )

    edge_info = {
        "Drug-DrugType":           int(mol_dt_edges.size(1)),
        "Drug-Disease":            int(mol_dis_edges.size(1)),
        "Drug-Gene":               int(mol_gene_edges.size(1)),
        "Gene-Reactome":           int(gene_rt_edges.size(1)),
        "Disease-TherapeuticArea": int(dis_ta_edges.size(1)),
        "Disease-Gene":            int(dis_gene_edges.size(1)),
    }
    print(f"  Edge info: {edge_info}")
    return edges, edge_info


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL VALIDATION / TEST SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def generate_val_test_splits(mappings, train_md_edges_set, approvedDrugs_array):
    """
    Replicate main_script logic:
      val  = new drug–disease edges in OT 23.06 that weren't in 21.06
      test = new drug–disease edges in OT 24.06 that weren't in 21.06 or 23.06
    """
    m = mappings

    def _load_indication_edges(path):
        dataset = ds.dataset(path, format="parquet")
        table   = dataset.to_table()
        expr    = pc.is_in(table.column("id"), value_set=approvedDrugs_array)
        table   = table.filter(expr)
        table   = table.select(["id","approvedIndications"]).flatten()
        return extract_test_edges(table, m["drug_key_mapping"], m["disease_key_mapping"])

    print("  Loading validation indication data (23.06) …")
    all_val_set  = _load_indication_edges(VAL_INDICATION_PATH)
    new_val_set  = all_val_set - train_md_edges_set
    print(f"  New validation edges: {len(new_val_set)}")

    print("  Loading test indication data (24.06) …")
    all_test_set = _load_indication_edges(TEST_INDICATION_PATH)
    new_test_set = all_test_set - train_md_edges_set - new_val_set
    print(f"  New test edges: {len(new_test_set)}")

    return new_val_set, new_test_set


def build_validation_tensors(new_val_set, not_linked_val, seed=42):
    random.seed(seed)
    true_pairs  = list(new_val_set)
    false_pairs = random.sample(not_linked_val, len(true_pairs))
    labels = [1]*len(true_pairs) + [0]*len(false_pairs)
    val_edges  = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
    val_labels = torch.tensor(labels, dtype=torch.long)
    return val_edges, val_labels, false_pairs


def build_test_tensors(new_test_set, not_linked_test, imbalance_ratio=1, seed=42):
    random.seed(seed)
    true_pairs  = list(new_test_set)
    num_neg     = min(len(true_pairs) * imbalance_ratio, len(not_linked_test))
    false_pairs = random.sample(not_linked_test, num_neg)
    labels = [1]*len(true_pairs) + [0]*len(false_pairs)
    test_edges  = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
    test_labels = torch.tensor(labels, dtype=torch.long)
    return test_edges, test_labels


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    mode = detect_data_mode()
    print(f"\nData mode: {mode}")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    if mode == "raw":
        (filtered_molecule_df, filtered_indication_df,
         filtered_disease_df, filtered_gene_table,
         gene_reactome_table, filtered_assoc_table) = load_raw_data()
    else:
        (filtered_molecule_df, filtered_indication_df,
         filtered_disease_df, filtered_gene_table,
         gene_reactome_table, filtered_assoc_table) = load_processed_data()

    # Convert to PyArrow for edge extraction
    filtered_molecule_table_pa   = pa.Table.from_pandas(filtered_molecule_df)
    filtered_indication_table_pa = pa.Table.from_pandas(filtered_indication_df)
    filtered_disease_table_pa    = pa.Table.from_pandas(filtered_disease_df)

    # approvedDrugs array needed for temporal split loading
    approvedDrugs_array = pa.array(filtered_molecule_df["id"].tolist())

    # ── 2. Node mappings ───────────────────────────────────────────────────────
    print("\nBuilding node mappings …")
    mappings = build_node_mappings(
        filtered_molecule_df, filtered_disease_df,
        filtered_gene_table, gene_reactome_table,
        filtered_molecule_table_pa
    )

    node_info = {
        "Drugs":               len(mappings["approved_drugs_list"]),
        "Drug_Types":          len(mappings["drug_type_list"]),
        "Genes":               len(mappings["gene_list"]),
        "Reactome_Pathways":   len(mappings["reactome_list"]),
        "Diseases":            len(mappings["disease_list"]),
        "Therapeutic_Areas":   len(mappings["therapeutic_area_list"]),
    }
    print(f"  Node info: {node_info}")

    # ── 3. Features ────────────────────────────────────────────────────────────
    print("\nBuilding node features …")
    all_features = build_features(
        filtered_molecule_table_pa, filtered_gene_table, mappings
    )

    # ── 4. Edges ───────────────────────────────────────────────────────────────
    print("\nBuilding edges …")
    edges, edge_info = build_edges(
        filtered_molecule_table_pa, filtered_indication_table_pa,
        filtered_disease_table_pa, gene_reactome_table,
        filtered_assoc_table, mappings
    )

    # ── 5. Training positive / negative pools ──────────────────────────────────
    print("\nBuilding negative sampling pools …")
    mol_dis_edges = edges["molecule_disease_edges"]
    train_md_edges_set = set(
        zip(mol_dis_edges[0].tolist(), mol_dis_edges[1].tolist())
    )

    all_pairs = [
        (mappings["drug_key_mapping"][d], mappings["disease_key_mapping"][dis])
        for d in mappings["approved_drugs_list"]
        for dis in mappings["disease_list"]
    ]
    print(f"  Total drug×disease pairs: {len(all_pairs):,}")

    # Save training edges CSV
    disease_offset = mappings["disease_offset"]
    training_rows = [
        (mappings["approved_drugs_list_name"][i],
         mappings["disease_list_name"][j - disease_offset])
        for i, j in train_md_edges_set
    ]
    pd.DataFrame(training_rows, columns=["drug_name","disease_name"]).to_csv(
        f"{RESULTS_PATH}training_drug_disease_pairs.csv", index=False
    )

    # ── 6. Temporal splits ─────────────────────────────────────────────────────
    print("\nGenerating temporal val/test splits …")
    try:
        new_val_set, new_test_set = generate_val_test_splits(
            mappings, train_md_edges_set, approvedDrugs_array
        )
    except Exception as e:
        print(f"  WARNING: Could not load temporal data ({e}). Using random splits.")
        not_linked_all = list(set(all_pairs) - train_md_edges_set)
        random.shuffle(not_linked_all)
        val_size  = min(len(train_md_edges_set) // 10, len(not_linked_all) // 4)
        test_size = val_size
        new_val_set  = set(not_linked_all[:val_size])
        new_test_set = set(not_linked_all[val_size:val_size+test_size])

    # Negative pools (no overlap with known positives)
    known_positives   = train_md_edges_set | new_val_set | new_test_set
    not_linked_all    = list(set(all_pairs) - train_md_edges_set)
    not_linked_val    = list(set(all_pairs) - train_md_edges_set - new_val_set)
    not_linked_test   = list(set(all_pairs) - known_positives)
    random.shuffle(not_linked_all)
    random.shuffle(not_linked_val)
    random.shuffle(not_linked_test)

    # Val tensors (balanced 1:1)
    val_edges, val_labels, _ = build_validation_tensors(new_val_set, not_linked_val)

    # Test tensors (balanced 1:1 for primary evaluation)
    test_edges, test_labels = build_test_tensors(new_test_set, not_linked_test, imbalance_ratio=1)

    # Fixed training negative edge index
    num_neg = len(train_md_edges_set)
    neg_edge_index = torch.tensor(
        random.sample(not_linked_all, min(num_neg, len(not_linked_all))),
        dtype=torch.long
    ).T  # [2, N]

    # Positive edge index for training
    pos_edge_index = torch.tensor(list(train_md_edges_set), dtype=torch.long).T  # [2, N]

    print(f"  Train pos: {pos_edge_index.size(1):,}, neg: {neg_edge_index.size(1):,}")
    print(f"  Val:  {val_edges.size(0):,} samples")
    print(f"  Test: {test_edges.size(0):,} samples")

    # Store positive sets as tensors for script 3 imbalance analysis
    val_pos_tensor  = torch.tensor(list(new_val_set),  dtype=torch.long)
    test_pos_tensor = torch.tensor(list(new_test_set), dtype=torch.long)

    # ── 7. Build PyG graph ────────────────────────────────────────────────────
    print("\nBuilding PyG Data graph …")
    all_edge_index = edges["all_edge_index"]
    edge_types     = edges["edge_types"]

    graph = Data(
        x=all_features,
        edge_index=all_edge_index,
        # Training tensors
        pos_edge_index=pos_edge_index,   # [2, N_pos]
        neg_edge_index=neg_edge_index,   # [2, N_neg]
        # Validation
        val_edge_index=val_edges,        # [N_val, 2]
        val_edge_label=val_labels,       # [N_val]
        # Test (balanced 1:1)
        test_edge_index=test_edges,      # [N_test, 2]
        test_edge_label=test_labels,     # [N_test]
        # Positive edge pools (for imbalance analysis in script 3)
        val_pos_edges=val_pos_tensor,    # [N_val_pos, 2]
        test_pos_edges=test_pos_tensor,  # [N_test_pos, 2]
        metadata={
            "node_info":     node_info,
            "edge_info":     edge_info,
            "data_mode":     mode,
            "training_version":   TRAINING_VERSION,
            "validation_version": VALIDATION_VERSION,
            "test_version":       TEST_VERSION,
            "as_dataset":         AS_DATASET,
            "neg_sampling":       NEG_SAMPLING,
            "disease_offset":     disease_offset,
            "num_relations":      6,
            "feature_columns":    GLOBAL_FEATURE_COLUMNS,
            "creation_timestamp": timestamp,
            "mappings_file":      f"{PROCESSED_PATH}mappings/all_mappings_{timestamp}.pkl",
        }
    )

    # Make undirected and attach edge_type (doubled to match undirected edges)
    graph = T.ToUndirected()(graph)
    edge_type_undirected = torch.cat([edge_types, edge_types], dim=0)
    assert edge_type_undirected.size(0) == graph.edge_index.size(1), (
        f"edge_type size {edge_type_undirected.size(0)} != "
        f"edge_index cols {graph.edge_index.size(1)}"
    )
    graph.edge_type = edge_type_undirected
    graph.x = graph.x.float()

    print(f"\nGraph validated: {graph.validate()}")
    print(graph)

    # ── 8. Save everything ────────────────────────────────────────────────────
    # Graph
    graph_path = f"{RESULTS_PATH}{TRAINING_VERSION}_{NEG_SAMPLING}_{AS_DATASET}_{timestamp}_graph.pt"
    torch.save(graph, graph_path)
    print(f"\nGraph saved → {graph_path}")

    # Edge tensors
    if mode == "raw":
        ed = f"{PROCESSED_PATH}edges/"
        torch.save(edges["molecule_drugType_edges"],   f"{ed}1_molecule_drugType_edges.pt")
        torch.save(edges["molecule_disease_edges"],    f"{ed}2_molecule_disease_edges.pt")
        torch.save(edges["molecule_gene_edges"],       f"{ed}3_molecule_gene_edges.pt")
        torch.save(edges["gene_reactome_edges"],       f"{ed}4_gene_reactome_edges.pt")
        torch.save(edges["disease_therapeutic_edges"],f"{ed}5_disease_therapeutic_edges.pt")
        torch.save(edges["disease_gene_edges"],        f"{ed}6_disease_gene_edges.pt")
        edge_stats = {**edge_info}
        with open(f"{ed}edge_statistics.json","w") as f:
            json.dump(edge_stats, f, indent=2)

    # Mappings – pickle (full, for downstream scripts)
    mappings_pkl = f"{PROCESSED_PATH}mappings/all_mappings_{timestamp}.pkl"
    with open(mappings_pkl, "wb") as f:
        pickle.dump(mappings, f)
    print(f"Mappings saved → {mappings_pkl}")

    # Mappings – individual JSON copies (human-readable)
    mp = f"{PROCESSED_PATH}mappings/"
    for key in ["drug_key_mapping","drug_type_key_mapping","gene_key_mapping",
                "reactome_key_mapping","disease_key_mapping","therapeutic_area_key_mapping"]:
        with open(f"{mp}{key}.json","w") as f:
            json.dump(mappings[key], f)

    # Mapping summary
    summary = {
        k: len(v) for k, v in node_info.items()
    }
    with open(f"{mp}mapping_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    # Processed tables (save for next run in processed mode)
    if mode == "raw":
        td = f"{PROCESSED_PATH}tables/"
        filtered_molecule_df.to_csv(f"{td}processed_molecules.csv", index=False)
        filtered_indication_df.to_csv(f"{td}processed_indications.csv", index=False)
        filtered_disease_df.to_csv(f"{td}processed_diseases.csv", index=False)
        filtered_gene_table.to_pandas().to_csv(f"{td}processed_genes.csv", index=False)
        filtered_assoc_table.to_pandas().to_csv(f"{td}processed_associations.csv", index=False)

    print("\n" + "="*60)
    print("GRAPH CREATION COMPLETE")
    print("="*60)
    print(f"  Nodes    : {graph.x.size(0):,}")
    print(f"  Edges    : {graph.edge_index.size(1):,}")
    print(f"  Features : {graph.x.size(1)}")
    print(f"  Val set  : {val_edges.size(0):,} samples")
    print(f"  Test set : {test_edges.size(0):,} samples")
    print(f"\nNext step:")
    print(f"  python scripts/2_train_models.py {graph_path} --mappings {mappings_pkl}")

    return graph_path, mappings_pkl


if __name__ == "__main__":
    main()
