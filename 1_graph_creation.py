"""
Graph Creation Module for Drug-Disease Prediction
This module handles data loading, preprocessing, and graph construction.
Supports both raw OpenTargets data (Option 1) and pre-processed data (Option 2).
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, HeteroData
import datetime as dt
import torch_geometric.transforms as T
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import numpy as np
import sys
import platform
import polars as pl
import ast
import os
import json
from torch_geometric.utils import to_networkx

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def enable_full_reproducibility(seed=42):
    """Enable full reproducibility with deterministic algorithms."""
    set_seed(seed)
    torch.use_deterministic_algorithms(True)

# Configuration
def get_config():
    """Get configuration parameters."""
    config = {
        'training_version': 21.06,
        'validation_version': 23.06,
        'test_version': 24.06,
        'as_dataset': 'associationByOverallDirect',
        'negative_sampling_approach': 'random',
        'general_path': 'data/raw/',
        'processed_path': 'data/processed/',
        'results_path': 'results/'
    }
    
    # Create proper directory paths
    os.makedirs(config['general_path'], exist_ok=True)
    os.makedirs(config['processed_path'], exist_ok=True)
    os.makedirs(config['results_path'], exist_ok=True)
    
    # Set specific paths for raw data (using forward slashes for cross-platform compatibility)
    config.update({
        'indication_path': f"{config['general_path']}{config['training_version']}/indication",
        'val_indication_path': f"{config['general_path']}{config['validation_version']}/indication",
        'test_indication_path': f"{config['general_path']}{config['test_version']}/indication",
        'molecule_path': f"{config['general_path']}{config['training_version']}/molecule",
        'disease_path': f"{config['general_path']}{config['training_version']}/diseases",
        'val_disease_path': f"{config['general_path']}{config['validation_version']}/diseases",
        'test_disease_path': f"{config['general_path']}{config['test_version']}/diseases",
        'gene_path': f"{config['general_path']}{config['training_version']}/targets",
        'associations_path': f"{config['general_path']}{config['training_version']}/{config['as_dataset']}"
    })
    
    return config

def detect_data_mode(config):
    """Detect whether to use raw data or pre-processed data."""
    processed_path = config['processed_path']
    
    # Check if pre-processed data exists (Option 2)
    processed_files_exist = (
        os.path.exists(f"{processed_path}tables/processed_molecules.csv") and
        os.path.exists(f"{processed_path}mappings/drug_key_mapping.json") and
        os.path.exists(f"{processed_path}edges/1_molecule_drugType_edges.pt")
    )
    
    # Check if raw data exists (Option 1)  
    raw_files_exist = (
        os.path.exists(config['indication_path']) and
        os.path.exists(config['molecule_path']) and
        os.path.exists(config['disease_path']) and
        os.path.exists(config['gene_path']) and
        os.path.exists(config['associations_path'])
    )
    
    if processed_files_exist:
        print("Pre-processed data detected - using Option 2 workflow (Quick Start)")
        print(f"Using pre-processed data from: {processed_path}")
        return "processed"
    elif raw_files_exist:
        print("Raw OpenTargets data detected - using Option 1 workflow (Complete Setup)")
        print(f"Using raw data from: {config['general_path']}")
        return "raw"
    else:
        print("ERROR: No valid data found!")
        print("\nPlease ensure you have either:")
        print(f"Option 1: Raw OpenTargets data in '{config['general_path']}'")
        print("   Required structure:")
        print("   ├── 21.06/")
        print("   │   ├── indication/")
        print("   │   ├── molecule/")
        print("   │   ├── diseases/")
        print("   │   ├── targets/")
        print("   │   └── associationByOverallDirect/")
        print("   ├── 23.06/indication/")
        print("   └── 24.06/indication/")
        print(f"\nOption 2: Pre-processed data in '{processed_path}'")
        print("   Required structure:")
        print("   ├── tables/processed_molecules.csv")
        print("   ├── mappings/drug_key_mapping.json")
        print("   └── edges/1_molecule_drugType_edges.pt")
        print("\nRefer to the README for detailed setup instructions.")
        
        raise FileNotFoundError(
            "Neither pre-processed nor raw data found. "
            "Please follow the README instructions for data preparation."
        )

# Utility Functions
def get_indices_from_keys(key_list, index_mapping):
    """Get indices from keys using mapping dictionary."""
    return [index_mapping[key] for key in key_list if key in index_mapping]

def generate_pairs(source_list, target_list, source_mapping, target_mapping, return_set=False, return_tensor=False):
    """Generate all possible edge combinations from 2 lists."""
    edges = []
    for source_id in source_list:
        for target_id in target_list:
            edges.append((source_mapping[source_id], target_mapping[target_id]))
    
    if return_set:
        return set(edges)
    elif return_tensor: 
        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    else: 
        return edges

def extract_edges(table, source_mapping, target_mapping, return_edge_list=False, return_edge_set=False):
    """Extract edges from a PyArrow table."""
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()
        target_list = targets.slice(i, 1).to_pylist()[0]
        
        if not isinstance(target_list, list):
            target_list = [target_list]
        
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))

    if return_edge_list:
        return edges
    elif return_edge_set:
        return set(edges)
    else:
        unique_edges = list(set(edges))
        edge_index_tensor = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor

# Feature Engineering Functions
def boolean_encode(boolean_array, pad_length):
    """Encode boolean arrays with padding."""
    boolean_series = pd.Series(boolean_array.to_pandas()).astype("float")
    boolean_array_filled = boolean_series.fillna(-1).to_numpy().reshape(-1, 1)
    tensor = torch.from_numpy(boolean_array_filled.astype(np.int64))

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def normalize(array, pad_length):
    """Normalize arrays with padding."""
    df = array.to_pandas().to_numpy().reshape(-1, 1)
    df = pd.DataFrame(df)
    df.fillna(-1, inplace=True)
    standardized = (df - df.mean()) / df.std()
    tensor = torch.from_numpy(standardized.to_numpy())

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def cat_encode(array, pad_length):
    """Encode categorical variables with padding."""
    uni = array.unique().to_pandas()
    unidict = {uni[i]: i for i in range(len(uni))}
    
    tensor = torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def pad_feature_matrix(matrix, pad_size, pad_value=-1):
    """Pad feature matrix to specified size."""
    if matrix.size(1) < pad_size:
        padding = torch.ones(matrix.size(0), pad_size - matrix.size(1)) * pad_value
        matrix = torch.cat([matrix, padding], dim=1)
    return matrix

def align_features(matrix, feature_columns, global_feature_columns):
    """Align feature matrices to global feature columns."""
    aligned_matrix = torch.zeros(matrix.size(0), len(global_feature_columns)) - 1  
    for idx, col in enumerate(feature_columns):
        global_idx = global_feature_columns.index(col)
        aligned_matrix[:, global_idx] = matrix[:, idx]
    return aligned_matrix

class GraphBuilder:
    """Main class for building the drug-disease prediction graph."""
    
    def __init__(self, config, data_mode):
        self.config = config
        self.data_mode = data_mode
        self.drug_key_mapping = {}
        self.disease_key_mapping = {}
        self.gene_key_mapping = {}
        self.drug_type_key_mapping = {}
        self.reactome_key_mapping = {}
        self.therapeutic_area_key_mapping = {}
        
    def load_data(self):
        """Load data based on the detected mode."""
        if self.data_mode == "processed":
            self.load_preprocessed_data()
        else:
            self.load_and_preprocess_raw_data()
    
    def load_preprocessed_data(self):
        """Load pre-processed data from files (Option 2)."""
        print("Loading pre-processed data files...")
        
        processed_path = self.config['processed_path']
        
        try:
            # Load processed tables
            print("   - Loading processed tables...")
            self.filtered_molecule_df = pd.read_csv(f"{processed_path}tables/processed_molecules.csv")
            self.filtered_indication_df = pd.read_csv(f"{processed_path}tables/processed_indications.csv")
            
            # Handle approvedIndications column that might be stored as strings
            if 'approvedIndications' in self.filtered_indication_df.columns:
                self.filtered_indication_df['approvedIndications'] = self.filtered_indication_df['approvedIndications'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                )
            
            self.filtered_disease_df = pd.read_csv(f"{processed_path}tables/processed_diseases.csv")
            self.filtered_gene_df = pd.read_csv(f"{processed_path}tables/processed_genes.csv")
            self.filtered_associations_df = pd.read_csv(f"{processed_path}tables/processed_associations.csv")
            
            # Load mappings
            print("   - Loading node mappings...")
            with open(f"{processed_path}mappings/drug_key_mapping.json", 'r') as f:
                self.drug_key_mapping = json.load(f)
            with open(f"{processed_path}mappings/drug_type_key_mapping.json", 'r') as f:
                self.drug_type_key_mapping = json.load(f)
            with open(f"{processed_path}mappings/gene_key_mapping.json", 'r') as f:
                self.gene_key_mapping = json.load(f)
            with open(f"{processed_path}mappings/reactome_key_mapping.json", 'r') as f:
                self.reactome_key_mapping = json.load(f)
            with open(f"{processed_path}mappings/disease_key_mapping.json", 'r') as f:
                self.disease_key_mapping = json.load(f)
            with open(f"{processed_path}mappings/therapeutic_area_key_mapping.json", 'r') as f:
                self.therapeutic_area_key_mapping = json.load(f)
            
            # Load edge tensors
            print("   - Loading pre-computed edges...")
            self.molecule_drugType_edges = torch.load(f"{processed_path}edges/1_molecule_drugType_edges.pt")
            self.molecule_disease_edges = torch.load(f"{processed_path}edges/2_molecule_disease_edges.pt")
            self.molecule_gene_edges = torch.load(f"{processed_path}edges/3_molecule_gene_edges.pt")
            self.gene_reactome_edges = torch.load(f"{processed_path}edges/4_gene_reactome_edges.pt")
            self.disease_therapeutic_edges = torch.load(f"{processed_path}edges/5_disease_therapeutic_edges.pt")
            self.disease_gene_edges = torch.load(f"{processed_path}edges/6_disease_gene_edges.pt")
            
            # Create node lists from mappings
            self.approved_drugs_list = list(self.drug_key_mapping.keys())
            self.drug_type_list = list(self.drug_type_key_mapping.keys())
            self.gene_list = list(self.gene_key_mapping.keys())
            self.reactome_list = list(self.reactome_key_mapping.keys())
            self.disease_list = list(self.disease_key_mapping.keys())
            self.therapeutic_area_list = list(self.therapeutic_area_key_mapping.keys())
            
            # Convert data to PyArrow tables for consistency with raw data workflow
            self.filtered_molecule_table = pa.Table.from_pandas(self.filtered_molecule_df)
            self.filtered_indication_table = pa.Table.from_pandas(self.filtered_indication_df)
            self.disease_table = pa.Table.from_pandas(self.filtered_disease_df)
            self.gene_table = pa.Table.from_pandas(self.filtered_gene_df)
            
            print("Pre-processed data loaded successfully!")
            print(f"   {len(self.approved_drugs_list)} drugs, {len(self.gene_list)} genes, {len(self.disease_list)} diseases")
            
        except Exception as e:
            raise FileNotFoundError(
                f"Error loading pre-processed data: {e}\n"
                "Please ensure you have run the data processing script to generate the required files."
            )
    
    def load_and_preprocess_raw_data(self):
        """Load and preprocess raw OpenTargets data (Option 1)."""
        print("Loading and preprocessing raw OpenTargets data...")
        
        try:
            # Load indication data
            print("   - Loading indication data...")
            indication_dataset = ds.dataset(self.config['indication_path'], format="parquet")
            indication_table = indication_dataset.to_table()
            
            # Filter for approved drugs
            expr = pc.list_value_length(pc.field("approvedIndications")) > 0 
            filtered_indication_table = indication_table.filter(expr)
            self.approvedDrugs = filtered_indication_table.column('id').combine_chunks()
            
            # Load and process molecule data
            print("   - Loading molecule data...")
            molecule_dataset = ds.dataset(self.config['molecule_path'], format="parquet")
            molecule_table = molecule_dataset.to_table()
            
            # Process drug types
            drug_type_column = pc.replace_substring(molecule_table.column('drugType'), 'unknown', 'Unknown')
            fill_value = pa.scalar('Unknown', type=pa.string())
            molecule_table = molecule_table.drop_columns("drugType").add_column(3, "drugType", drug_type_column.fill_null(fill_value))
            
            # Apply redundant ID mappings
            print("   - Applying ID mappings...")
            self.filtered_molecule_df, self.filtered_indication_df = self._apply_id_mappings(
                molecule_table, filtered_indication_table
            )
            
            # Load gene data
            print("   - Loading gene/target data...")
            gene_dataset = ds.dataset(self.config['gene_path'], format="parquet")
            self.gene_table = gene_dataset.to_table().flatten().flatten()
            
            # Load disease data
            print("   - Loading disease data...")
            disease_dataset = ds.dataset(self.config['disease_path'], format="parquet")
            self.disease_table = self._preprocess_disease_data(disease_dataset.to_table())
            
            # Load associations data
            print("   - Loading associations data...")
            associations_dataset = ds.dataset(self.config['associations_path'], format="parquet")
            self.associations_table = self._preprocess_associations_data(associations_dataset.to_table())
            
            # Create node mappings for raw data
            print("   - Creating node mappings...")
            self.create_node_mappings_from_raw()
            
            print("Raw data loading and preprocessing completed!")
            print(f"   {len(self.approved_drugs_list)} drugs, {len(self.gene_list)} genes, {len(self.disease_list)} diseases")
            
        except Exception as e:
            print(f"ERROR: Error loading raw data: {e}")
            print("\nPlease ensure you have:")
            print("1. Downloaded the required OpenTargets datasets")
            print("2. Placed them in the correct directory structure as per README")
            print("3. Renamed 'disease' to 'diseases' and 'target' to 'targets'")
            raise
        
    def _apply_id_mappings(self, molecule_table, filtered_indication_table):
        """Apply redundant ID mappings for consistency."""
        # Define redundant mappings
        redundant_id_mapping = {
            'CHEMBL1200538': 'CHEMBL632',
            'CHEMBL1200376': 'CHEMBL632',
            'CHEMBL1200384': 'CHEMBL632',
            'CHEMBL1201207': 'CHEMBL632',
            'CHEMBL1497': 'CHEMBL632',
            'CHEMBL1201661': 'CHEMBL3989767',
            'CHEMBL1506': 'CHEMBL130',
            'CHEMBL1201281': 'CHEMBL130',
            'CHEMBL1201289': 'CHEMBL1753',
            'CHEMBL3184512': 'CHEMBL1753',
            'CHEMBL1530428': 'CHEMBL384467',
            'CHEMBL1201302': 'CHEMBL384467',
            'CHEMBL1511': 'CHEMBL135',
            'CHEMBL4298187': 'CHEMBL2108597',
            'CHEMBL4298110': 'CHEMBL2108597',
            'CHEMBL1200640': 'CHEMBL2108597',
            'CHEMBL989': 'CHEMBL1501',
            'CHEMBL1201064': 'CHEMBL1200600',
            'CHEMBL1473': 'CHEMBL1676',
            'CHEMBL1201512': 'CHEMBL1201688',
            'CHEMBL1201657': 'CHEMBL1201513',
            'CHEMBL1091': 'CHEMBL389621',
            'CHEMBL1549': 'CHEMBL389621',
            'CHEMBL3989663': 'CHEMBL389621',
            'CHEMBL1641': 'CHEMBL389621',
            'CHEMBL1200562': 'CHEMBL389621',
            'CHEMBL1201544': 'CHEMBL2108597',
            'CHEMBL1200823': 'CHEMBL2108597',
            'CHEMBL2021423': 'CHEMBL1200572',
            'CHEMBL1364144':'CHEMBL650',
            'CHEMBL1200844': 'CHEMBL650',
            'CHEMBL1201265': 'CHEMBL650',
            'CHEMBL1140': 'CHEMBL573',
            'CHEMBL1152': 'CHEMBL131',
            'CHEMBL1201231': 'CHEMBL131',
            'CHEMBL1200909': 'CHEMBL131',
            'CHEMBL635': 'CHEMBL131',
            'CHEMBL1200335': 'CHEMBL386630',
            'CHEMBL1504': 'CHEMBL1451',
            'CHEMBL1200449': 'CHEMBL1451',
            'CHEMBL1200878': 'CHEMBL1451',
            'CHEMBL1200929': 'CHEMBL3988900'
        }
        
        redundant_id_mapping_D = {
            'EFO_1000905': 'EFO_0004228',
            'EFO_0005752': 'EFO_1001888',
            'EFO_0007512': 'EFO_0007510'
        }
        
        # Process molecule data
        filtered_molecule_table = molecule_table.select([
            'id', 'name', 'drugType', 'blackBoxWarning', 'yearOfFirstApproval',
            'parentId', 'childChemblIds', 'linkedDiseases', 'hasBeenWithdrawn', 'linkedTargets'
        ]).flatten().drop_columns(['linkedTargets.count', 'linkedDiseases.count'])
        
        filtered_molecule_df = filtered_molecule_table.to_pandas()
        filtered_molecule_df = filtered_molecule_df[pd.isna(filtered_molecule_df['parentId'])]
        
        # Process indication data
        filtered_indication_df = filtered_indication_table.to_pandas()
        
        # Apply disease ID mappings (simplified implementation)
        def safe_list_conversion(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except:
                    return []
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, list):
                return value
            return [value]

        def update_approved_indications(disease_list, mapping_dict):
            if not isinstance(disease_list, list):
                return disease_list
            return [mapping_dict.get(str(d), str(d)) for d in disease_list]

        filtered_indication_df['approvedIndications'] = filtered_indication_df['approvedIndications'].apply(safe_list_conversion)
        filtered_indication_df['approvedIndications'] = filtered_indication_df['approvedIndications'].apply(
            lambda x: update_approved_indications(x, redundant_id_mapping_D)
        )
        
        # Filter molecules to only include those with approved indications
        unique_chembl_ids = filtered_indication_df['id'].unique()
        filtered_molecule_df = filtered_molecule_df[filtered_molecule_df['id'].isin(unique_chembl_ids)]
        
        return filtered_molecule_df, filtered_indication_df
    
    def _preprocess_disease_data(self, disease_table):
        """Preprocess disease data."""
        # Filter out unwanted therapeutic areas
        disease_table = disease_table.filter(pc.list_value_length(pc.field("therapeuticAreas")) > 0)
        df = disease_table.to_pandas()
        filtered_df = df[~df['therapeuticAreas'].apply(lambda x: 'EFO_0001444' in x)]
        disease_table = pa.Table.from_pandas(filtered_df)
        
        # Select relevant columns
        disease_table = disease_table.select([
            'id', 'name', 'description', 'ancestors', 'descendants', 'children', 'therapeuticAreas'
        ])
        
        # Filter out unwanted prefixes
        prefixes_to_remove = ["UBERON", "ZFA", "CL", "GO", "FBbt", "FMA"]
        filter_conditions = [pc.starts_with(disease_table.column('id'), prefix) for prefix in prefixes_to_remove]
        
        combined_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_filter = pc.or_(combined_filter, condition)
        
        negated_filter = pc.invert(combined_filter)
        filtered_disease_table = disease_table.filter(negated_filter)
        
        # Additional filtering
        filtered_disease_table = filtered_disease_table.filter(pc.list_value_length(pc.field("descendants")) == 0)
        filtered_disease_table = filtered_disease_table.filter(pc.field("id") != "EFO_0000544")
        
        return filtered_disease_table
    
    def _preprocess_associations_data(self, associations_table):
        """Preprocess associations data."""
        # Find score column
        score_column = None
        for col in associations_table.column_names:
            if "Score" in col or "score" in col:
                score_column = col
                break
        
        associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
        return associations_table
    
    def create_node_mappings_from_raw(self):
        """Create node mappings from raw data."""
        print("Creating node mappings from raw data...")
        
        # Extract unique node lists
        self.approved_drugs_list = list(self.filtered_molecule_df['id'].unique())
        self.gene_list = list(self.gene_table.column('id').unique().to_pylist())
        self.disease_list = list(self.disease_table.column('id').unique().to_pylist())
        
        drug_type = self.filtered_molecule_df['drugType'].dropna().unique().tolist()
        self.drug_type_list = drug_type
        
        # Extract reactome and therapeutic areas
        if self.config['training_version'] == 21.04 or self.config['training_version'] == 21.06:
            filtered_gene_table = self.gene_table.select(['id', 'approvedName','bioType', 'proteinAnnotations.functions', 'reactome']).flatten()
            reactome = filtered_gene_table.column('reactome').combine_chunks().flatten()
        else:
            filtered_gene_table = self.gene_table.select(['id', 'approvedName','biotype', 'functionDescriptions', 'proteinIds', 'pathways']).flatten()
            reactome = filtered_gene_table.column('pathways').combine_chunks().flatten()
            reactome = reactome.field(0)
        
        self.reactome_list = list(reactome.unique().to_pylist())
        
        therapeutic_area = self.disease_table.column('therapeuticAreas').combine_chunks().flatten()
        self.therapeutic_area_list = list(therapeutic_area.unique().to_pylist())
        
        # Create mappings
        self.drug_key_mapping = {self.approved_drugs_list[i]: i for i in range(len(self.approved_drugs_list))}
        
        offset = len(self.drug_key_mapping)
        self.drug_type_key_mapping = {self.drug_type_list[i]: i + offset for i in range(len(self.drug_type_list))}
        
        offset += len(self.drug_type_key_mapping)
        self.gene_key_mapping = {self.gene_list[i]: i + offset for i in range(len(self.gene_list))}
        
        offset += len(self.gene_key_mapping)
        self.reactome_key_mapping = {self.reactome_list[i]: i + offset for i in range(len(self.reactome_list))}
        
        offset += len(self.reactome_key_mapping)
        self.disease_key_mapping = {self.disease_list[i]: i + offset for i in range(len(self.disease_list))}
        
        offset += len(self.disease_key_mapping)
        self.therapeutic_area_key_mapping = {self.therapeutic_area_list[i]: i + offset for i in range(len(self.therapeutic_area_list))}
        
        print(f"Created mappings for {len(self.drug_key_mapping)} drugs, {len(self.gene_key_mapping)} genes, {len(self.disease_key_mapping)} diseases")
    
    def create_features(self):
        """Create node features."""
        print("Creating node features...")
        
        # Get indices for different node types
        drug_indices = torch.tensor(get_indices_from_keys(self.approved_drugs_list, self.drug_key_mapping), dtype=torch.long)
        drug_type_indices = torch.tensor(get_indices_from_keys(self.drug_type_list, self.drug_type_key_mapping), dtype=torch.long)
        gene_indices = torch.tensor(get_indices_from_keys(self.gene_list, self.gene_key_mapping), dtype=torch.long)
        reactome_indices = torch.tensor(get_indices_from_keys(self.reactome_list, self.reactome_key_mapping), dtype=torch.long)
        disease_indices = torch.tensor(get_indices_from_keys(self.disease_list, self.disease_key_mapping), dtype=torch.long)
        therapeutic_area_indices = torch.tensor(get_indices_from_keys(self.therapeutic_area_list, self.therapeutic_area_key_mapping), dtype=torch.long)
        
        # Create drug features
        if self.data_mode == "processed":
            # For pre-processed data, use the dataframe directly
            molecule_table = self.filtered_molecule_table
        else:
            # For raw data, convert dataframe to PyArrow table
            molecule_table = pa.Table.from_pandas(self.filtered_molecule_df)
        
        blackBoxWarning = molecule_table.column('blackBoxWarning').combine_chunks()
        blackBoxWarning_vector = boolean_encode(blackBoxWarning, drug_indices)
        
        yearOfFirstApproval = molecule_table.column('yearOfFirstApproval').combine_chunks()
        yearOfFirstApproval_vector = normalize(yearOfFirstApproval, drug_indices)
        
        # Create one-hot encodings for different node types
        drug_one_hot = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        drug_type_one_hot = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        gene_one_hot = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        reactome_one_hot = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        disease_one_hot = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        therapeutic_area_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        
        # Create feature matrices for each node type
        drug_node_type_vector = torch.tensor([drug_one_hot], dtype=torch.float32).repeat(len(drug_indices), 1)
        drug_feature_matrix = torch.cat((drug_node_type_vector, blackBoxWarning_vector, yearOfFirstApproval_vector), dim=1)
        
        drug_type_node_type_vector = torch.tensor([drug_type_one_hot], dtype=torch.float32).repeat(len(drug_type_indices), 1)
        drug_type_feature_matrix = torch.cat((drug_type_node_type_vector, torch.ones(len(drug_type_indices), 2) * -1), dim=1)
        
        gene_node_type_vector = torch.tensor([gene_one_hot], dtype=torch.float32).repeat(len(gene_indices), 1)
        gene_feature_matrix = torch.cat((gene_node_type_vector, torch.ones(len(gene_indices), 2) * -1), dim=1)
        
        reactome_node_type_vector = torch.tensor([reactome_one_hot], dtype=torch.float32).repeat(len(reactome_indices), 1)
        reactome_feature_matrix = torch.cat((reactome_node_type_vector, torch.ones(len(reactome_indices), 2) * -1), dim=1)
        
        disease_node_type_vector = torch.tensor([disease_one_hot], dtype=torch.float32).repeat(len(disease_indices), 1)
        disease_feature_matrix = torch.cat((disease_node_type_vector, torch.ones(len(disease_indices), 2) * -1), dim=1)
        
        therapeutic_area_node_type_vector = torch.tensor([therapeutic_area_one_hot], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1)
        therapeutic_area_feature_matrix = torch.cat((therapeutic_area_node_type_vector, torch.ones(len(therapeutic_area_indices), 2) * -1), dim=1)
        
        # Combine all feature matrices
        self.all_features = torch.cat([
            drug_feature_matrix,
            drug_type_feature_matrix,
            gene_feature_matrix,
            reactome_feature_matrix,
            disease_feature_matrix,
            therapeutic_area_feature_matrix
        ], dim=0)
        
        print(f"Created feature matrix with shape: {self.all_features.shape}")_type_vector, torch.ones(len(disease_indices), 2) * -1), dim=1)
        
        therapeutic_area_node_type_vector = torch.tensor([therapeutic_area_one_hot], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1)
        therapeutic_area_feature_matrix = torch.cat((therapeutic_area_node_type_vector, torch.ones(len(therapeutic_area_indices), 2) * -1), dim=1)
        
        # Combine all feature matrices
        self.all_features = torch.cat([
            drug_feature_matrix,
            drug_type_feature_matrix,
            gene_feature_matrix,
            reactome_feature_matrix,
            disease_feature_matrix,
            therapeutic_area_feature_matrix
        ], dim=0)
        
        print(f"Created feature matrix with shape: {self.all_features.shape}")
    
    def create_edges(self):
        """Create edge indices for the graph."""
        print("Creating graph edges...")
        
        if self.data_mode == "processed":
            # For pre-processed data, edges are already loaded
            print("   Using pre-processed edge tensors")
            all_edges = [
                self.molecule_drugType_edges,
                self.molecule_disease_edges,
                self.molecule_gene_edges,
                self.gene_reactome_edges,
                self.disease_therapeutic_edges,
                self.disease_gene_edges
            ]
        else:
            # For raw data, extract edges from tables
            print("   Extracting edges from raw data...")
            
            # Convert dataframes back to PyArrow tables
            filtered_molecule_table = pa.Table.from_pandas(self.filtered_molecule_df)
            filtered_indication_table = pa.Table.from_pandas(self.filtered_indication_df)
            
            # Extract different edge types
            print("     - Drug-DrugType edges...")
            molecule_drugType_table = filtered_molecule_table.select(['id', 'drugType']).drop_null().flatten()
            self.molecule_drugType_edges = extract_edges(molecule_drugType_table, self.drug_key_mapping, self.drug_type_key_mapping)
            self.molecule_drugType_edges = torch.unique(self.molecule_drugType_edges, dim=1)
            
            print("     - Drug-Disease edges...")
            molecule_disease_table = filtered_indication_table.select(['id', 'approvedIndications']).flatten()
            self.molecule_disease_edges = extract_edges(molecule_disease_table, self.drug_key_mapping, self.disease_key_mapping)
            self.molecule_disease_edges = torch.unique(self.molecule_disease_edges, dim=1)
            
            print("     - Drug-Gene edges...")
            molecule_gene_table = filtered_molecule_table.select(['id', 'linkedTargets.rows']).drop_null().flatten()
            self.molecule_gene_edges = extract_edges(molecule_gene_table, self.drug_key_mapping, self.gene_key_mapping)
            self.molecule_gene_edges = torch.unique(self.molecule_gene_edges, dim=1)
            
            print("     - Gene-Reactome edges...")
            # Create gene-reactome edges based on version
            if self.config['training_version'] == 21.04 or self.config['training_version'] == 21.06:
                gene_reactome_table = self.gene_table.select(['id', 'reactome']).flatten()
            else:
                gene_reactome_df = self.gene_table.select(['id', 'pathways']).flatten().to_pandas()
                exploded = gene_reactome_df.explode('pathways')
                exploded['pathwayId'] = exploded['pathways'].apply(lambda x: x['pathwayId'] if pd.notnull(x) else None)
                final_df = exploded[['id', 'pathwayId']]
                gene_reactome_table = pa.Table.from_pandas(final_df).drop_null()
            
            self.gene_reactome_edges = extract_edges(gene_reactome_table, self.gene_key_mapping, self.reactome_key_mapping)
            self.gene_reactome_edges = torch.unique(self.gene_reactome_edges, dim=1)
            
            print("     - Disease-Therapeutic edges...")
            disease_therapeutic_table = self.disease_table.select(['id', 'therapeuticAreas']).drop_null().flatten()
            self.disease_therapeutic_edges = extract_edges(disease_therapeutic_table, self.disease_key_mapping, self.therapeutic_area_key_mapping)
            self.disease_therapeutic_edges = torch.unique(self.disease_therapeutic_edges, dim=1)
            
            print("     - Disease-Gene edges...")
            disease_gene_table = self.associations_table.select(['diseaseId', 'targetId']).flatten()
            self.disease_gene_edges = extract_edges(disease_gene_table, self.disease_key_mapping, self.gene_key_mapping)
            self.disease_gene_edges = torch.unique(self.disease_gene_edges, dim=1)
            
            all_edges = [
                self.molecule_drugType_edges,
                self.molecule_disease_edges,
                self.molecule_gene_edges,
                self.gene_reactome_edges,
                self.disease_therapeutic_edges,
                self.disease_gene_edges
            ]
        
        # Combine all edges
        self.all_edge_index = torch.cat(all_edges, dim=1)
        
        print(f"Created {self.all_edge_index.size(1)} total edges:")
        print(f"   - Drug -> DrugType: {all_edges[0].size(1):,}")
        print(f"   - Drug -> Disease: {all_edges[1].size(1):,}")
        print(f"   - Drug -> Gene: {all_edges[2].size(1):,}")
        print(f"   - Gene -> Reactome: {all_edges[3].size(1):,}")
        print(f"   - Disease -> Therapeutic: {all_edges[4].size(1):,}")
        print(f"   - Disease -> Gene: {all_edges[5].size(1):,}")(lambda x: x['pathwayId'] if pd.notnull(x) else None)
                final_df = exploded[['id', 'pathwayId']]
                gene_reactome_table = pa.Table.from_pandas(final_df).drop_null()
            
            self.gene_reactome_edges = extract_edges(gene_reactome_table, self.gene_key_mapping, self.reactome_key_mapping)
            self.gene_reactome_edges = torch.unique(self.gene_reactome_edges, dim=1)
            
            disease_therapeutic_table = self.disease_table.select(['id', 'therapeuticAreas']).drop_null().flatten()
            self.disease_therapeutic_edges = extract_edges(disease_therapeutic_table, self.disease_key_mapping, self.therapeutic_area_key_mapping)
            self.disease_therapeutic_edges = torch.unique(self.disease_therapeutic_edges, dim=1)
            
            disease_gene_table = self.associations_table.select(['diseaseId', 'targetId']).flatten()
            self.disease_gene_edges = extract_edges(disease_gene_table, self.disease_key_mapping, self.gene_key_mapping)
            self.disease_gene_edges = torch.unique(self.disease_gene_edges, dim=1)
            
            all_edges = [
                self.molecule_drugType_edges,
                self.molecule_disease_edges,
                self.molecule_gene_edges,
                self.gene_reactome_edges,
                self.disease_therapeutic_edges,
                self.disease_gene_edges
            ]
        
        # Combine all edges
        self.all_edge_index = torch.cat(all_edges, dim=1)
        
        print(f"Created {self.all_edge_index.size(1)} total edges")
        print(f"  - Drug-DrugType: {all_edges[0].size(1)}")
        print(f"  - Drug-Disease: {all_edges[1].size(1)}")
        print(f"  - Drug-Gene: {all_edges[2].size(1)}")
        print(f"  - Gene-Reactome: {all_edges[3].size(1)}")
        print(f"  - Disease-Therapeutic: {all_edges[4].size(1)}")
        print(f"  - Disease-Gene: {all_edges[5].size(1)}")
    
    def create_validation_test_splits(self):
        """Create validation and test edge splits."""
        print("Creating validation and test splits...")
        
        # For validation data
        val_indication_dataset = ds.dataset(self.config['val_indication_path'], format="parquet")
        val_indication_table = val_indication_dataset.to_table()
        
        # Filter validation data for approved drugs
        approved_drugs_array = pa.array(self.approved_drugs_list)
        expr1 = pc.is_in(val_indication_table.column('id'), value_set=approved_drugs_array)
        val_filtered_indication_table = val_indication_table.filter(expr1)
        val_molecule_disease_table = val_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        
        # Extract validation edges
        all_val_md_edges_set = extract_edges(val_molecule_disease_table, self.drug_key_mapping, self.disease_key_mapping, return_edge_set=True)
        
        # Get training edges
        if self.data_mode == "processed":
            train_md_edges_set = set(zip(self.molecule_disease_edges[0].tolist(), self.molecule_disease_edges[1].tolist()))
        else:
            train_md_edges_set = extract_edges(
                pa.Table.from_pandas(self.filtered_indication_df).select(['id', 'approvedIndications']).flatten(), 
                self.drug_key_mapping, self.disease_key_mapping, return_edge_set=True
            )
        
        # Find new validation edges
        self.new_val_edges_set = all_val_md_edges_set - train_md_edges_set
        
        # Create negative samples for validation
        all_molecule_disease = generate_pairs(self.approved_drugs_list, self.disease_list, 
                                            self.drug_key_mapping, self.disease_key_mapping)
        not_linked_set = list(set(all_molecule_disease) - train_md_edges_set)
        
        # Create validation tensors
        true_pairs = list(self.new_val_edges_set)
        random.seed(42)
        false_pairs = random.sample(not_linked_set, len(true_pairs))
        
        true_labels = [1] * len(true_pairs)
        false_labels = [0] * len(false_pairs)
        combined_labels = true_labels + false_labels
        
        self.val_edge_tensor = torch.tensor(true_pairs + false_pairs, dtype=torch.long)
        self.val_label_tensor = torch.tensor(combined_labels, dtype=torch.long)
        
        # Create test data similarly
        test_indication_dataset = ds.dataset(self.config['test_indication_path'], format="parquet")
        test_indication_table = test_indication_dataset.to_table()
        
        expr2 = pc.is_in(test_indication_table.column('id'), value_set=approved_drugs_array)
        test_filtered_indication_table = test_indication_table.filter(expr2)
        test_molecule_disease_table = test_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        
        all_test_md_edges_set = extract_edges(test_molecule_disease_table, self.drug_key_mapping, self.disease_key_mapping, return_edge_set=True)
        self.new_test_edges_set = all_test_md_edges_set - train_md_edges_set - all_val_md_edges_set
        
        # Create test tensors
        test_true_pairs = list(self.new_test_edges_set)
        test_false_pairs = random.sample([pair for pair in not_linked_set if pair not in false_pairs], len(test_true_pairs))
        
        test_true_labels = [1] * len(test_true_pairs)
        test_false_labels = [0] * len(test_false_pairs)
        test_combined_labels = test_true_labels + test_false_labels
        
        self.test_edge_tensor = torch.tensor(test_true_pairs + test_false_pairs, dtype=torch.long)
        self.test_label_tensor = torch.tensor(test_combined_labels, dtype=torch.long)
        
        print(f"Created validation set with {len(true_pairs)} positive and {len(false_pairs)} negative samples")
        print(f"Created test set with {len(test_true_pairs)} positive and {len(test_false_pairs)} negative samples")
    
    def build_graph(self):
        """Build the complete graph object."""
        print("Building final graph structure...")
        
        # Create metadata
        node_info = {
            "Drugs": len(self.approved_drugs_list),
            "Drug_Types": len(self.drug_type_list),
            "Genes": len(self.gene_list),
            "Reactome_Pathways": len(self.reactome_list),
            "Diseases": len(self.disease_list),
            "Therapeutic_Areas": len(self.therapeutic_area_list)
        }
        
        edge_info = {
            "Drug-DrugType": int(self.molecule_drugType_edges.size(1)),
            "Drug-Disease": int(self.molecule_disease_edges.size(1)),
            "Drug-Gene": int(self.molecule_gene_edges.size(1)),
            "Gene-Reactome": int(self.gene_reactome_edges.size(1)),
            "Disease-Therapeutic": int(self.disease_therapeutic_edges.size(1)),
            "Disease-Gene": int(self.disease_gene_edges.size(1))
        }
        
        metadata = {
            "node_info": node_info,
            "edge_info": edge_info,
            "data_mode": self.data_mode,
            "config": self.config,
            "creation_timestamp": dt.datetime.now().isoformat(),
            "total_nodes": sum(node_info.values()),
            "total_edges": sum(edge_info.values())
        }
        
        # Create graph
        graph = Data(
            x=self.all_features, 
            edge_index=self.all_edge_index,
            val_edge_index=self.val_edge_tensor,
            val_edge_label=self.val_label_tensor,
            test_edge_index=self.test_edge_tensor,
            test_edge_label=self.test_label_tensor,
            metadata=metadata
        )
        
        # Convert to undirected
        graph = T.ToUndirected()(graph)
        
        print("Graph construction completed!")
        print(f"   Nodes: {graph.x.size(0):,} | Edges: {graph.edge_index.size(1):,}")
        print(f"   Features: {graph.x.size(1)} dimensions")
        print(f"   Validation samples: {len(self.val_edge_tensor):,}")
        print(f"   Test samples: {len(self.test_edge_tensor):,}")
        
        return graph

def create_graph(config=None):
    """Main function to create the graph."""
    if config is None:
        config = get_config()
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Detect data mode
    print("Detecting available data...")
    data_mode = detect_data_mode(config)
    
    # Initialize graph builder
    builder = GraphBuilder(config, data_mode)
    
    # Build graph step by step
    print(f"\nBuilding graph using {data_mode} data...")
    builder.load_data()
    builder.create_features()
    builder.create_edges()
    builder.create_validation_test_splits()
    
    # Build final graph
    graph = builder.build_graph()
    
    # Save graph with descriptive filename
    datetime_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_mode_suffix = "preprocessed" if data_mode == "processed" else "raw"
    graph_filename = f"graph_{config['training_version']}_{data_mode_suffix}_{datetime_str}.pt"
    graph_path = os.path.join(config['results_path'], graph_filename)
    
    torch.save(graph, graph_path)
    
    print(f"\nGraph creation completed!")
    print(f"Graph saved to: {graph_path}")
    print(f"Data mode used: Option {'2' if data_mode == 'processed' else '1'} ({data_mode})")
    print(f"Total nodes: {graph.x.size(0):,}")
    print(f"Total edges: {graph.edge_index.size(1):,}")
    print(f"Feature dimensions: {graph.x.size(1)}")
    
    return graph, graph_path, builder

def load_config_from_file(config_path="config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # Get default config and update with file config
        config = get_config()
        config.update(file_config)
        
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using default configuration")
        return get_config()
    except json.JSONDecodeError:
        print(f"Invalid JSON in {config_path}, using default configuration")
        return get_config()

if __name__ == "__main__":
    print("Drug-Disease Prediction - Graph Creation")
    print("=" * 50)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config_from_file(config_path)
        print(f"Using configuration from: {config_path}")
    else:
        config = load_config_from_file()  # Try to load config.json by default
        print("Using default configuration")
    
    print(f"Data paths:")
    print(f"   Raw data: {config.get('general_path', 'N/A')}")
    print(f"   Processed data: {config.get('processed_path', 'N/A')}")
    print(f"   Results: {config.get('results_path', 'N/A')}")
    
    graph, graph_path, builder = create_graph(config)
