"""
Graph Creation Module for Drug-Disease Prediction
This module handles data loading, preprocessing, and graph construction.
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
        'disease_similarity_network': False,
        'molecule_similarity_network': False,
        'reactome_network': False,
        'trial_edges': False,
        'negative_sampling_approach': 'random'
    }
    
    # Platform-specific paths
    if platform.system() == "Windows":
        config.update({
            'general_path': r"C:\\OpenTargets_datasets\\downloads\\",
            'results_path': r"C:\\OpenTargets_datasets\\test_results3\\",
            'dict_path': r"C:\\OpenTargets_datasets\\test_results_biosb\\"
        })
    else:
        config.update({
            'general_path': "OT/",
            'results_path': "test_results/",
            'dict_path': "test_results_biosb/"
        })
    
    # Set specific paths
    config.update({
        'indication_path': f"{config['general_path']}{config['training_version']}\\indication",
        'val_indication_path': f"{config['general_path']}{config['validation_version']}\\indication",
        'test_indication_path': f"{config['general_path']}{config['test_version']}\\indication",
        'molecule_path': f"{config['general_path']}{config['training_version']}\\molecule",
        'disease_path': f"{config['general_path']}{config['training_version']}\\diseases",
        'val_disease_path': f"{config['general_path']}{config['validation_version']}\\diseases",
        'test_disease_path': f"{config['general_path']}{config['test_version']}\\diseases",
        'gene_path': f"{config['general_path']}{config['training_version']}\\targets",
        'associations_path': f"{config['general_path']}{config['training_version']}/{config['as_dataset']}"
    })
    
    return config

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
    
    def __init__(self, config):
        self.config = config
        self.drug_key_mapping = {}
        self.disease_key_mapping = {}
        self.gene_key_mapping = {}
        self.drug_type_key_mapping = {}
        self.reactome_key_mapping = {}
        self.therapeutic_area_key_mapping = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets."""
        print("Loading and preprocessing data...")
        
        # Load indication data
        indication_dataset = ds.dataset(self.config['indication_path'], format="parquet")
        indication_table = indication_dataset.to_table()
        
        # Filter for approved drugs
        expr = pc.list_value_length(pc.field("approvedIndications")) > 0 
        filtered_indication_table = indication_table.filter(expr)
        self.approvedDrugs = filtered_indication_table.column('id').combine_chunks()
        
        # Load and process molecule data
        molecule_dataset = ds.dataset(self.config['molecule_path'], format="parquet")
        molecule_table = molecule_dataset.to_table()
        
        # Process drug types
        drug_type_column = pc.replace_substring(molecule_table.column('drugType'), 'unknown', 'Unknown')
        fill_value = pa.scalar('Unknown', type=pa.string())
        molecule_table = molecule_table.drop_columns("drugType").add_column(3, "drugType", drug_type_column.fill_null(fill_value))
        
        # Apply redundant ID mappings
        self.filtered_molecule_df, self.filtered_indication_df = self._apply_id_mappings(
            molecule_table, filtered_indication_table
        )
        
        # Load gene data
        gene_dataset = ds.dataset(self.config['gene_path'], format="parquet")
        self.gene_table = gene_dataset.to_table().flatten().flatten()
        
        # Load disease data
        disease_dataset = ds.dataset(self.config['disease_path'], format="parquet")
        self.disease_table = self._preprocess_disease_data(disease_dataset.to_table())
        
        # Load associations data
        associations_dataset = ds.dataset(self.config['associations_path'], format="parquet")
        self.associations_table = self._preprocess_associations_data(associations_dataset.to_table())
        
        print("Data loading and preprocessing completed.")
        
    def _apply_id_mappings(self, molecule_table, filtered_indication_table):
        """Apply redundant ID mappings for consistency."""
        # Define redundant mappings
        redundant_id_mapping = {
            'CHEMBL1200538': 'CHEMBL632',
            'CHEMBL1200376': 'CHEMBL632',
            # Add other mappings as needed...
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
        
        # Apply mappings (simplified for brevity)
        # In practice, you'd implement the full mapping logic here
        
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
        
        if self.config['training_version'] == 21.04:
            associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
        else:
            associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
        
        return associations_table
    
    def create_node_mappings(self):
        """Create node mappings for all node types."""
        print("Creating node mappings...")
        
        # Extract unique node lists
        self.approved_drugs_list = list(self.filtered_molecule_df['id'].unique())
        self.gene_list = list(self.gene_table.column('id').unique().to_pylist())
        self.disease_list = list(self.disease_table.column('id').unique().to_pylist())
        
        drug_type = self.filtered_molecule_df['drugType'].dropna().unique().tolist()
        self.drug_type_list = drug_type
        
        # Create mappings
        self.drug_key_mapping = {self.approved_drugs_list[i]: i for i in range(len(self.approved_drugs_list))}
        
        offset = len(self.drug_key_mapping)
        self.drug_type_key_mapping = {self.drug_type_list[i]: i + offset for i in range(len(self.drug_type_list))}
        
        offset += len(self.drug_type_key_mapping)
        self.gene_key_mapping = {self.gene_list[i]: i + offset for i in range(len(self.gene_list))}
        
        # Continue for other node types...
        
        print(f"Created mappings for {len(self.drug_key_mapping)} drugs, {len(self.gene_key_mapping)} genes, {len(self.disease_key_mapping)} diseases")
    
    def create_features(self):
        """Create node features."""
        print("Creating node features...")
        
        # Define node types
        num_node_types = 6
        
        # Get indices
        drug_indices = torch.tensor(get_indices_from_keys(self.approved_drugs_list, self.drug_key_mapping), dtype=torch.long)
        gene_indices = torch.tensor(get_indices_from_keys(self.gene_list, self.gene_key_mapping), dtype=torch.long)
        disease_indices = torch.tensor(get_indices_from_keys(self.disease_list, self.disease_key_mapping), dtype=torch.long)
        
        # Create drug features
        filtered_molecule_table = pa.Table.from_pandas(self.filtered_molecule_df)
        blackBoxWarning = filtered_molecule_table.column('blackBoxWarning').combine_chunks()
        blackBoxWarning_vector = boolean_encode(blackBoxWarning, drug_indices)
        
        yearOfFirstApproval = filtered_molecule_table.column('yearOfFirstApproval').combine_chunks()
        yearOfFirstApproval_vector = normalize(yearOfFirstApproval, drug_indices)
        
        drug_one_hot = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        drug_node_type_vector = torch.tensor([drug_one_hot], dtype=torch.float32).repeat(len(drug_indices), 1)
        
        drug_feature_matrix = torch.cat((drug_node_type_vector, blackBoxWarning_vector, yearOfFirstApproval_vector), dim=1)
        
        # Create other feature matrices (simplified)
        # ... implement similar logic for other node types
        
        # Pad and align all features
        pad_size = 9
        global_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot', 'blackBoxWarning', 'yearOfFirstApproval', 'bioType']
        
        # Apply padding and alignment to all feature matrices
        drug_feature_matrix = pad_feature_matrix(drug_feature_matrix, pad_size, -1)
        
        # Stack all features
        self.all_features = drug_feature_matrix  # Simplified - add other features in practice
        
        print(f"Created feature matrix with shape: {self.all_features.shape}")
    
    def create_edges(self):
        """Create edge indices for the graph."""
        print("Creating edges...")
        
        # Convert dataframes back to PyArrow tables
        filtered_molecule_table = pa.Table.from_pandas(self.filtered_molecule_df)
        filtered_indication_table = pa.Table.from_pandas(self.filtered_indication_df)
        
        # Extract different edge types
        molecule_disease_table = filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        self.molecule_disease_edges = extract_edges(molecule_disease_table, self.drug_key_mapping, self.disease_key_mapping)
        
        # Create other edge types (simplified for brevity)
        # In practice, implement all edge extractions here
        
        # Combine all edges
        self.all_edge_index = self.molecule_disease_edges  # Simplified
        
        print(f"Created {self.all_edge_index.size(1)} edges")
    
    def create_validation_test_splits(self):
        """Create validation and test edge splits."""
        print("Creating validation and test splits...")
        
        # Load validation data
        val_indication_dataset = ds.dataset(self.config['val_indication_path'], format="parquet")
        val_indication_table = val_indication_dataset.to_table()
        
        # Extract validation edges
        expr1 = pc.is_in(val_indication_table.column('id'), value_set=self.approvedDrugs)
        val_filtered_indication_table = val_indication_table.filter(expr1)
        val_molecule_disease_table = val_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
        
        all_val_md_edges_set = extract_edges(val_molecule_disease_table, self.drug_key_mapping, self.disease_key_mapping, return_edge_set=True)
        train_md_edges_set = extract_edges(pa.Table.from_pandas(self.filtered_indication_df).select(['id', 'approvedIndications']).flatten(), 
                                         self.drug_key_mapping, self.disease_key_mapping, return_edge_set=True)
        
        self.new_val_edges_set = all_val_md_edges_set - train_md_edges_set
        
        # Create negative samples
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
        
        print(f"Created validation set with {len(true_pairs)} positive and {len(false_pairs)} negative samples")
    
    def build_graph(self):
        """Build the complete graph object."""
        print("Building final graph...")
        
        # Create metadata
        node_info = {
            "Drugs": len(self.approved_drugs_list),
            "Genes": len(self.gene_list),
            "Diseases": len(self.disease_list)
        }
        
        metadata = {"node_info": node_info}
        
        # Create graph
        graph = Data(
            x=self.all_features, 
            edge_index=self.all_edge_index,
            val_edge_index=self.val_edge_tensor,
            val_edge_label=self.val_label_tensor,
            metadata=metadata
        )
        
        # Convert to undirected
        graph = T.ToUndirected()(graph)
        
        print("Graph creation completed!")
        print(f"Graph: {graph}")
        
        return graph

def create_graph(config=None):
    """Main function to create the graph."""
    if config is None:
        config = get_config()
    
    # Set reproducibility
    enable_full_reproducibility(42)
    
    # Initialize graph builder
    builder = GraphBuilder(config)
    
    # Build graph step by step
    builder.load_and_preprocess_data()
    builder.create_node_mappings()
    builder.create_features()
    builder.create_edges()
    builder.create_validation_test_splits()
    
    # Build final graph
    graph = builder.build_graph()
    
    # Save graph
    datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    graph_path = f"{config['results_path']}{config['training_version']}_{config['negative_sampling_approach']}_{config['as_dataset']}_{datetime_str}_graph.pt"
    torch.save(graph, graph_path)
    
    print(f"Graph saved to: {graph_path}")
    
    return graph, graph_path, builder

if __name__ == "__main__":
    graph, graph_path, builder = create_graph()