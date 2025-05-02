import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, HeteroData
import datetime as dt
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import random
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv
import seaborn as sns
import logging
import numpy as np
import sys
import platform
import polars as pl
import itertools
from sklearn.metrics import average_precision_score
import plotly.graph_objects as go
import ast
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def enable_full_reproducibility(seed=42):
    set_seed(seed)
    torch.use_deterministic_algorithms(True)

# Set up full reproducibility
enable_full_reproducibility(42)


dict_path = r"C:\\OpenTargets_datasets\\test_results_biosb\\"
training_version = 21.06
validation_version = 23.06
test_version = 24.06


#Chose the model
model_choice = TransformerConv #GCNConv #TransformerConv #SAGEConv


as_dataset = 'associationByOverallDirect'
disease_similarity_network = False
molecule_similarity_network = False
reactome_network = False
trial_edges= False
#negative_sampling_approach = f"BPR" #BPR #f"AS {as_dataset}" #random #BPR_AS
negative_sampling_approach = f"random"

model_choices=("GCNConv" "SAGEConv" "TransformerConv")
as_datasets=("associationByDatasourceDirect" "associationByDatasourceIndirect" "associationByDatatypeDirect" "associationByDatatypeIndirect" "associationByOverallDirect" "associationByOverallIndirect")
negative_sampling_methods=("random", f"AS {as_dataset}", "BPR", f"BPR_{as_dataset}")



# Detect the operating system and define paths accordingly
if platform.system() == "Windows":
    general_path = r"C:\\OpenTargets_datasets\\downloads\\"
    results_path = r"C:\\OpenTargets_datasets\\test_results3\\"
    indication_path = f"{general_path}{training_version}\\indication"
    val_indication_path = f"{general_path}{validation_version}\\indication"
    test_indication_path = f"{general_path}{test_version}\\indication"
    molecule_path = f"{general_path}{training_version}\\molecule"
    disease_path = f"{general_path}{training_version}\\diseases"
    val_disease_path = f"{general_path}{validation_version}\\diseases"
    test_disease_path = f"{general_path}{test_version}\\diseases"
    gene_path = f"{general_path}{training_version}\\targets"
    associations_path = f"{general_path}{training_version}/{as_dataset}"

else:
    general_path = "OT/"
    results_path = "test_results/"
    indication_path = f"{general_path}{training_version}/indication"
    val_indication_path = f"{general_path}{validation_version}/indication"
    test_indication_path = f"{general_path}{test_version}\\indication"
    molecule_path = f"{general_path}{training_version}/molecule"
    disease_path = f"{general_path}{training_version}/diseases"
    val_disease_path = f"{general_path}{validation_version}/diseases"
    test_disease_path = f"{general_path}{test_version}/diseases"
    gene_path = f"{general_path}{training_version}/targets"
    associations_path = f"{general_path}{training_version}/{as_dataset}"


#Function to get indices from keys
def get_indices_from_keys(key_list, index_mapping):
    return [index_mapping[key] for key in key_list if key in index_mapping]

#Function to generate all possible edge combinations from 2 lists
def generate_pairs(source_list, target_list, source_mapping, target_mapping, return_set=False, return_tensor=False):
    edges = []
    for source_id in source_list:
        for target_id in target_list:
            edges.append((source_mapping[source_id], target_mapping[target_id]))
    if return_set:
        return set(edges)
    elif return_tensor: 
        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    else: return edges
  

#Function to generate tensor for all possible edge combinations from 2 lists
def generate_tensor(source_list, target_list, source_mapping, target_mapping):
    edges = []
    for i in range(len(source_list)):
        edges.append((source_mapping[source_list[i]], target_mapping[target_list[i]]))
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index_tensor


# Function to extract edges from a table
def extract_edges(table, source_mapping, target_mapping, return_edge_list=False, return_edge_set=False):
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()  # Get the individual node ID
        target_list = targets.slice(i, 1).to_pylist()[0]  # Extract the target list for this source
        #Ensure that target_list is actually a list before iterating
        if not isinstance(target_list, list):
            target_list = [target_list]
        # Create a pair for each target and append it to the edges list
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
                

    if return_edge_list:
        return edges
    elif return_edge_set:
        return set(edges)
    else:
        unique_edges = list(set(edges))  # Deduplicate
        edge_index_tensor = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        return edge_index_tensor
    
def extract_edges_no_mapping(table, return_edge_list=False, return_edge_set=False):
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()  # Get the individual source node ID
        target_list = targets.slice(i, 1).to_pylist()[0]  # Extract the target list for this source
        
        # Ensure that target_list is actually a list before iterating
        if not isinstance(target_list, list):
            target_list = [target_list]
        
        # Create a string representation for each edge and append it to the edges list
        for target_id in target_list:
            edges.append(f"{source_id} -> {target_id}")
    
    if return_edge_list:
        return edges  # Return the list of edges as strings
    elif return_edge_set:
        return set(edges)  # Return the set of unique edges
    else:
        return edges  # Default to returning the list of edges

    
def extract_test_edges(table, source_mapping, target_mapping):
    source = table.column(0).combine_chunks()
    targets = table.column(1).combine_chunks()
  
    edges = []
    for i in range(len(source)):
        source_id = source[i].as_py()  # Get the individual node ID
        target_list = targets.slice(i, 1).to_pylist()[0]  # Extract the target list for this source
        #Ensure that target_list is actually a list before iterating
        if not isinstance(target_list, list):
            target_list = [target_list]
        # Create a pair for each target and append it to the edges list
        for target_id in target_list:
            if source_id in source_mapping and target_id in target_mapping:
                edges.append((source_mapping[source_id], target_mapping[target_id]))
                
    return set(edges)
   
#Function to find repurposing edges
def find_repurposing_edges(table1, table2, column_name, source_mapping, target_mapping):
    # Create a mask for filtering: True if element of table2['id'] is in table1['id']
    filter_mask = pc.is_in(table2.column('id'), value_set=table1.column('id'))
    # Use the filter Function to apply the mask to table2
    filtered_table = table2.filter(filter_mask)
    # Create new edge list
    new_edge_list = []
    for i in range(len(filtered_table)):
        row = filtered_table.slice(i, 1)
        drug_id = row.column('id').combine_chunks()[0].as_py()
        linked_items = row.column(column_name).combine_chunks()[0].as_py()

        for item in linked_items:
            if drug_id in source_mapping and item in target_mapping:
                new_edge_list.append((source_mapping[drug_id], target_mapping[item]))
    return new_edge_list




def generate_edge_sets_val(training_version, validation_version):
    # Validation tensor extraction
    val_indication_dataset = ds.dataset(val_indication_path, format="parquet")
    val_indication_table = val_indication_dataset.to_table()
    # Filter for approved drugs from training version
    expr1 = pc.is_in(val_indication_table.column('id'), value_set=approvedDrugs)
    val_filtered_indication_table = val_indication_table.filter(expr1)
    val_molecule_disease_table = val_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
    all_val_md_edges_set = extract_edges(val_molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_set=True)
    print("Validation set total:", len(all_val_md_edges_set))

    # Extract training edges
    train_md_edges_set = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_set=True)
    print("Training set:", len(train_md_edges_set))


    # Determine new validation edges (exclude training edges)
    new_val_edges_set = all_val_md_edges_set - train_md_edges_set
    print("Validation set new:", len(new_val_edges_set))
  
    # Convert the set of tuples to a list of tuples
    new_val_edges_set_edges = list(new_val_edges_set)
    print("Existing drug disease edges:", len(existing_drug_disease_edges))

    # Convert the list of tuples to a list of strings
    new_val_edges_set_edges_str = [(approved_drugs_list_name[i], disease_list_name[j-3644]) for i, j in new_val_edges_set_edges]

    # Save the new_val_edges_set_edges to a CSV file
    new_val_edges_set_edges_str_df = pd.DataFrame(new_val_edges_set_edges_str, columns=['drug_name', 'disease_name'])
    new_val_edges_set_edges_str_df.to_csv(f"{dict_path}new_validation_drug_disease_edges_name.csv")
    # save as an xlsx file
    new_val_edges_set_edges_str_df.to_excel(f"{dict_path}new_validation_drug_disease_edges_name.xlsx")

    return train_md_edges_set, new_val_edges_set
def create_validation_tensors(new_val_edges_set, not_linked_set_validation):
    # True pairs
    true_pairs = list(new_val_edges_set)
    seed = 42
    random.seed(seed)
    # Sample false pairs for validation
    false_pairs = random.sample(not_linked_set_validation, len(true_pairs))
    print("Number of true pairs:", len(true_pairs))
    print("Number of false pairs:", len(false_pairs))

    # Create labels
    true_labels = [1] * len(true_pairs)
    false_labels = [0] * len(false_pairs)

    # Combine labels
    combined_labels = true_labels + false_labels
    label_tensor = torch.tensor(combined_labels, dtype=torch.long)

    # Create the validation edge tensor
    val_edge_tensor = torch.tensor(true_pairs + false_pairs, dtype=torch.long)

    return val_edge_tensor, label_tensor, false_pairs


def generate_edge_sets_test(training_version, validation_version, test_version, val_edge_tensor):
    test_indication_dataset = ds.dataset(test_indication_path, format="parquet")
    test_indication_table = test_indication_dataset.to_table()
    
    # Filter for approved drugs from training version
    expr1 = pc.is_in(test_indication_table.column('id'), value_set=approvedDrugs)
    test_filtered_indication_table = test_indication_table.filter(expr1)
    test_molecule_disease_table = test_filtered_indication_table.select(['id', 'approvedIndications']).flatten()
    
    test_md_edges_set = extract_test_edges(test_molecule_disease_table, drug_key_mapping, disease_key_mapping)
    print("Test set total:", len(test_md_edges_set))
    
    new_test_edges = test_md_edges_set - val_edge_tensor - train_md_edges_set
    print("Test set new:", len(new_test_edges))

     # Convert the set of tuples to a list of tuples
    new_test_edges_set = list(new_test_edges)


    # Convert the list of tuples to a list of strings
    new_test_edges_str = [(approved_drugs_list_name[i], disease_list_name[j-3644]) for i, j in new_test_edges_set]

    # Save the new_val_edges_set_edges to a CSV file
    new_test_edges_str_df = pd.DataFrame(new_test_edges_str, columns=['drug_name', 'disease_name'])
    new_test_edges_str_df.to_csv(f"{dict_path}new_test_drug_disease_edges_name2.csv")
    # save as an xlsx file
    new_test_edges_str_df.to_excel(f"{dict_path}new_test_drug_disease_edges_name2.xlsx")

    
    return new_test_edges


def create_test_tensors(new_test_edges_set, not_linked_set_test):
    # Fix randomness for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    # Generate Cartesian product
    true_pairs = list(new_test_edges_set)  # True pairs

    false_pairs = random.sample(not_linked_set_test, len(true_pairs)) #### balanced dataset

    # Create labels
    true_labels = [1] * len(true_pairs)
    false_labels = [0] * len(false_pairs)
    print("Number of true pairs in test:", len(true_pairs))
    print("Number of false pairs in test:", len(false_pairs))
    # Combine labels
    combined_labels = true_labels + false_labels

    # Create label tensor
    label_tensor = torch.tensor(combined_labels, dtype=torch.long)

    # Ensure the test tensor is aligned with labels
    test_edge_tensor = torch.tensor(list(true_pairs) + list(false_pairs), dtype=torch.long)  # Convert sets back to lists
    return test_edge_tensor, label_tensor

def boolean_encode(boolean_array, pad_length):
    # Convert to Pandas Series (ensure float to handle NaNs correctly)
    boolean_series = pd.Series(boolean_array.to_pandas()).astype("float")
    # Convert to NumPy array and fill NaNs with -1
    boolean_array_filled = boolean_series.fillna(-1).to_numpy().reshape(-1, 1)

    # Convert to PyTorch tensor (ensure int64 type)
    tensor = torch.from_numpy(boolean_array_filled.astype(np.int64))

    # Calculate padding size
    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor


def normalize(array, pad_length):
    df = array.to_pandas().to_numpy().reshape(-1, 1)
    df = pd.DataFrame(df)  # Explicitly create a new DataFrame

    df.fillna(-1, inplace=True)  # Modify in-place for efficiency

    standardized = (df - df.mean()) / df.std()

    tensor = torch.from_numpy(standardized.to_numpy())

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

def cat_encode(array, pad_length):
    uni = array.unique().to_pandas()
    unidict = {uni[i]: i for i in range(len(uni))}
    
    tensor = torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

    max_length = len(pad_length)
    padding_size = max_length - tensor.shape[0]

    # Pad the tensor
    if padding_size > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, padding_size), value=-1)
    else:
        padded_tensor = tensor

    return padded_tensor

# #Function to encode categorical variables
# def cat_encode(array):
#     uni = array.unique().to_pandas()
#     unidict = {uni[i]: i for i in range(len(uni))}
        
#     return torch.tensor([unidict[i] for i in array.to_pandas()], dtype=torch.int32)

#Function to generate word embeddings
def word_embeddings(array):
    array = [text if text is not None else "" for text in array.to_pylist()]
    batch_size = 32
    embeddings_list = []
    # load the tokenizer and model, and call the Function
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    for i in tqdm(range(0, len(array), batch_size), desc="Processing batches"):
        batch_texts = array[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        embeddings = output.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    
    return embeddings_tensor

def one_hot_encode(node_type):
    feature_vector = torch.zeros(num_node_types)

    feature_vector[node_type] = 1
    return feature_vector.unsqueeze(0)
'''
def pad_feature_matrix_with_mask(matrix, pad_size):
    mask = torch.zeros(matrix.size(0), pad_size)
    if matrix.size(1) < pad_size:
        padding = torch.zeros(matrix.size(0), pad_size - matrix.size(1))
        matrix = torch.cat([matrix, padding], dim=1)
    mask[:, :matrix.size(1)] = 1
    return matrix, mask
'''
def pad_feature_matrix(matrix, pad_size, pad_value=-1):

    if matrix.size(1) < pad_size:
        padding = torch.ones(matrix.size(0), pad_size - matrix.size(1)) * -1  # Fill with -1
        matrix = torch.cat([matrix, padding], dim=1)
    return matrix

#Function to generate tensor for all possible edge combinations from 2 lists
def generate_edge_list(source_list, target_list, source_mapping, target_mapping):
   edges = []
   for i in range(len(source_list)):
       edges.append((source_mapping[source_list[i]], target_mapping[target_list[i]]))
   return edges


def as_negative_sampling(filtered_molecule_table, associations_table, score_column, return_list=False, return_set=False):
  
   # Table with Molecule and linked targets
   MLT = filtered_molecule_table.select(['id', 'linkedTargets.rows']).drop_null()

   # Table with Disease and linked targets and association scores
   DT = associations_table.select(['diseaseId', 'targetId', score_column])
   print("DT:", DT.slice(0,2).to_pandas(), len(DT)) 


   logging.info("Table with Disease and linked targets:", DT.slice(0,2).to_pandas(), len(DT))

   logging.info("Table with Molecule and linked targets:", MLT.slice(0,2).to_pandas(), len(MLT))

   #Convert to pandas DataFrames
   df_DT = DT.to_pandas()
   df_MLT= MLT.to_pandas()

   # Explode the 'linkedTargets.rows' column to create separate rows for each molecule, target pair
   df_MLT_exploded = df_MLT.explode('linkedTargets.rows').reset_index(drop=True)
   # Rename the column to 'targetId' for key matching during join
   df_MLT_exploded.rename(columns={'linkedTargets.rows': 'targetId'}, inplace=True)
   logging.info("Exploded Table with Molecule and linked targets:", df_MLT_exploded.head(2), len(df_MLT_exploded))
   MLT_exploded= pa.Table.from_pandas(df_MLT_exploded)

   # Convert MLT_exploded and DT to Polars DataFrames
   pl_MLT_exploded = pl.from_pandas(df_MLT_exploded)
   pl_DT = pl.from_pandas(df_DT)

   # Initialize an empty Polars DataFrame
   final_df = pl.DataFrame()

    # Memory management for big files
   if len(DT) > 1000000:
    # Divide the table into slices of 1 million rows each
    for i in tqdm(range(0, len(DT), 1000000), desc="Processing chunks with polars"):
        # Slice the tables
        t_DT = DT.slice(i, min(i + 1000000, len(DT)))
        MTD_table1 = MLT_exploded.slice(i, min(i + 1000000, len(MLT_exploded)))
        
        # Convert chunks to Polars DataFrames
        pl_t_DT = pl.from_arrow(t_DT)
        pl_MTD_table1 = pl.from_arrow(MTD_table1)
        
        # Join the tables on 'targetId'
        joined_chunk = pl_MTD_table1.join(pl_t_DT, on='targetId')
        
        # Sort the joined chunk by score_column
        sorted_chunk = joined_chunk.sort(score_column)
        
        # Concatenate the chunk to final_df
        final_df = pl.concat([final_df, sorted_chunk], how='vertical')
        MTD_table = final_df.to_arrow()
        print("Final table length:", len(MTD_table))

#    #Malloc Memory management for big files
#    if len(DT) > 1000000: 
#    #divide the table into slices fo 1 million rows each 
#        for i in tqdm(range(0, len(DT), 1000000), desc="Processing chunks without polars"):
#            t_DT = DT.slice(i, i+1000000)
#            MTD_table1 = MLT_exploded.slice(i, i+1000000)
#            MTD_table1 = MLT_exploded.join(t_DT, 'targetId').combine_chunks().sort_by(score_column)
#            # Create an empty table with the same schema as MTD_table1
#            empty_table_schema = MTD_table1.schema
#            MTD_table2 = pa.Table.from_arrays([pa.array([])] * len(empty_table_schema.names), schema=empty_table_schema)
#            # Concatenate the tables
#            MTD_table = pa.concat_tables([MTD_table1, MTD_table2])
#            print("Final table length:", len(MTD_table))
   
   else:
       MTD_table = MLT_exploded.join(DT, 'targetId').combine_chunks().sort_by(score_column)

   #Optional: Drop columns that are not needed for negative sampling
   logging.info("Table with Molecule, Disease and association scores:", MTD_table.to_pandas().head(2), len(MTD_table))
   expr = pc.field(score_column) <= pc.scalar(0.01001)
   logging.info(MTD_table.filter(expr).to_pandas().head(5))
   #Drop other columns to make a negative sample table
   negative_sample_table = MTD_table.drop_columns(['targetId',score_column]).drop_null()
   mlist = negative_sample_table.column('id').combine_chunks().to_pylist()
   dlist = negative_sample_table.column('diseaseId').combine_chunks().to_pylist()

   #create edge list
   ng_list = generate_edge_list(mlist, dlist, drug_key_mapping, disease_key_mapping)

   logging.info('Negative list created:', ng_list[0:5])
   if return_list:
         return ng_list
   elif return_set:   return set(ng_list)
   else: return torch.tensor(ng_list, dtype=torch.long).t().contiguous()


def custom_edges(disease_similarity_network, trial_edges, molecule_similarity_network,
                       filtered_disease_table, filtered_molecule_table,
                       disease_key_mapping, drug_key_mapping):

    # Initialize an empty list to collect all active edges
    custom_edges = []

    # Condition for disease similarity network edges
    if disease_similarity_network:
        disease_descendants_table = filtered_disease_table.select(['id', 'descendants']).flatten()
        disease_children_table = filtered_disease_table.select(['id', 'children']).flatten()
        disease_ancestors_table = filtered_disease_table.select(['id', 'ancestors']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(disease_descendants_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_children_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(disease_ancestors_table, disease_key_mapping, disease_key_mapping, return_edge_list=True))

        # Extract unique disease nodes 
        # disease_descendants = filtered_disease_table.column('descendants').combine_chunks().flatten().unique()
        # disease_children = filtered_disease_table.column('children').combine_chunks().flatten().unique()
        # disease_ancestors = filtered_disease_table.column('ancestors').combine_chunks().flatten().unique()

        # # Update disease key mapping with new nodes if do not already exist in the mapping
        # all_disease_ids = set(disease_descendants) | set(disease_children) | set(disease_ancestors)

        # for disease_id in all_disease_ids:
        #     disease_key_mapping.setdefault(disease_id, len(disease_key_mapping)) 


    # Condition for trial edges
    if trial_edges:
        molecule_trial_table = filtered_molecule_table.select(['id', 'linkedDiseases']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(molecule_trial_table, drug_key_mapping, disease_key_mapping, return_edge_list=True))

        #update disease key mapping with new nodes if do not already exist in the mapping
        # trial_diseases = molecule_trial_table.column('linkedDiseases').combine_chunks().flatten().unique()

        # # Update disease key mapping with new nodes if do not already exist in the mapping
        # for disease_id in trial_diseases:
        #     disease_key_mapping.setdefault(disease_id, len(disease_key_mapping))      

    # Condition for molecule similarity network edges
    if molecule_similarity_network:
        molecule_parents_table = filtered_molecule_table.select(['id', 'parentId']).flatten()

        molecule_children_table = filtered_molecule_table.select(['id', 'childChemblIds']).flatten()

        # Extract edges and add to all_edges
        custom_edges.extend(extract_edges(molecule_parents_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))
        custom_edges.extend(extract_edges(molecule_children_table, drug_key_mapping, drug_key_mapping, return_edge_list=True))

        #Extract unique molecule nodes
        # molecule_parents = filtered_molecule_table.column('parentId').combine_chunks().flatten().unique()
        # molecule_children = filtered_molecule_table.column('childChemblIds').combine_chunks().flatten().unique()
        
        # # Update drug key mapping with new nodes if do not already exist in the mapping
        # all_molecule_ids = set(molecule_parents) | set(molecule_children)
        # for molecule_id in all_molecule_ids:
        #     drug_key_mapping.setdefault(molecule_id, len(drug_key_mapping))
    custom_edge_tensor = torch.tensor(custom_edges, dtype=torch.long).t().contiguous()

    return custom_edge_tensor





# extract nodes from each dataset
indication_dataset = ds.dataset(indication_path, format="parquet")
indication_table = indication_dataset.to_table()

expr = pc.list_value_length(pc.field("approvedIndications")) > 0 
filtered_indication_table = indication_table.filter(expr)

approvedDrugs = filtered_indication_table.column('id').combine_chunks()

# get the pair between the approved drugs and the indications, the indication column is a list
approvedDrugsIndications = filtered_indication_table.select(['id', 'approvedIndications']).flatten()
# save the approved drugs and indications to a csv file
approvedDrugsIndications.to_pandas().to_csv(f"{results_path}approvedDrugsIndications.csv", index=False)


#approvedIndications = filtered_indication_table.column('approvedIndications').combine_chunks()
#unique_approved_indications = approvedIndications.flatten().unique()




molecule_dataset = ds.dataset(molecule_path, format="parquet")
molecule_table = molecule_dataset.to_table()


molecule_drugType_table = molecule_table.select(['id', 'drugType'])

#Replace 'unknown' with 'Unknown'
drug_type_column = pc.replace_substring(molecule_drugType_table[1], 'unknown', 'Unknown')
#Replace null with 'Unknown'
fill_value = pa.scalar('Unknown', type = pa.string())

molecule_table = molecule_table.drop_columns("drugType").add_column(3,"drugType", drug_type_column.fill_null(fill_value))

molecule_drugType_table = molecule_table.select(['id', 'drugType'])


all_moleculesin = molecule_table.column('id').combine_chunks()
'''
#Filter for molecules that are present in the filtered indication dataset
molecule_filter = pc.is_in(molecule_table.column('id'), value_set= pc.unique(approvedDrugs))
filtered_molecule_table = molecule_table.filter(molecule_filter)
'''
filtered_molecule_table = molecule_table.select(['id','name','drugType','blackBoxWarning','yearOfFirstApproval','parentId', 'childChemblIds', 'linkedDiseases', 'hasBeenWithdrawn', 'linkedTargets']).flatten().drop_columns(['linkedTargets.count', 'linkedDiseases.count'])

# Convert PyArrow table to Pandas DataFrame
filtered_molecule_df = filtered_molecule_table.to_pandas()
# use approvedDrugs to filter the filtered_molecule_df
#filtered_molecule_df = filtered_molecule_df[filtered_molecule_df['id'].isin(approvedDrugs.to_pandas())]



# Step 1: Create a mapping dictionary from filtered_molecule_df
id_to_parentid_mapping = filtered_molecule_df.set_index('id')['parentId'].to_dict()

# Remove rows with a `parentId` in `filtered_molecule_df`
filtered_molecule_df = filtered_molecule_df[pd.isna(filtered_molecule_df['parentId'])]

# Save the updated table to a new CSV file
output_path = f'{results_path}updated_filtered_molecule_table1217.csv'
filtered_molecule_df.to_csv(output_path, index=False)
# save as an xlsx file


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

def resolve_mapping(chembl_id, mapping_dict):
    """Recursively resolve ID mappings to the final target."""
    visited = set()  # Prevent infinite loops
    while chembl_id in mapping_dict and chembl_id not in visited:
        visited.add(chembl_id)
        chembl_id = mapping_dict[chembl_id]
    return chembl_id

# Function to map disease IDs inside deeply nested structures
# Function to replace disease IDs inside lists
# Function to replace disease IDs inside lists
# Step 1: Ensure list format
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

# Step 2: Replace IDs inside the list
def update_approved_indications(disease_list, mapping_dict):
    if not isinstance(disease_list, list):
        return disease_list
    return [mapping_dict.get(str(d), str(d)) for d in disease_list]


# Step 1: Apply the updated redundant ID mapping
id_to_parentid_mapping = {k: resolve_mapping(v, redundant_id_mapping) for k, v in redundant_id_mapping.items()}

print(id_to_parentid_mapping)
# Debugging: Print a few resolved mappings
print("Sample resolved mappings (first 10):")
for i, (k, v) in enumerate(id_to_parentid_mapping.items()):
    print(f"{k} → {v}")
    if i >= 9:  # Print only first 10 to avoid too much output
        break

# Step 3: Update IDs in filtered_indication_df based on the new mapping
filtered_indication_df = filtered_indication_table.to_pandas()

filtered_indication_df['id'] = filtered_indication_df['id'].apply(
    lambda x: resolve_mapping(x, id_to_parentid_mapping) if x in id_to_parentid_mapping else x
)

# Debugging: Print before replacing approvedIndications
print("Before updating approvedIndications:")
print(filtered_indication_df[['id', 'approvedIndications']].head())

redundant_id_mapping_D = {
    'EFO_1000905': 'EFO_0004228',
    'EFO_0005752': 'EFO_1001888',
    'EFO_0007512': 'EFO_0007510'}

# Step 3: Apply both steps
filtered_indication_df['approvedIndications'] = filtered_indication_df['approvedIndications'].apply(safe_list_conversion)
filtered_indication_df['approvedIndications'] = filtered_indication_df['approvedIndications'].apply(
    lambda x: update_approved_indications(x, redundant_id_mapping_D)
)

# Debugging: Print after replacing approvedIndications
print("After updating approvedIndications:")
print(filtered_indication_df[['id', 'approvedIndications']].head())

# Convert the filtered indication DataFrame back to a PyArrow Table
filtered_indication_table_mapped = pa.Table.from_pandas(filtered_indication_df)


# Verify that mappings were applied correctly
print("Final Check: Does EFO_0007512 still exist?")
print(filtered_indication_df[filtered_indication_df['approvedIndications'].astype(str).str.contains('EFO_0007512')])
print(filtered_indication_df.loc[filtered_indication_df['id'].isin(['CHEMBL1200493', 'CHEMBL2105689', 'CHEMBL626']), 'approvedIndications'])


# Get unique CHEMBL IDs
unique_chembl_ids = filtered_indication_df[filtered_indication_df.columns[0]].unique()

print("Unique CHEMBL IDs:")
print(unique_chembl_ids)
print("Number of unique CHEMBL IDs:", len(unique_chembl_ids))


# Filter the molecule table to only keep rows where id is in unique_chembl_ids
filtered_molecule_df = filtered_molecule_df[filtered_molecule_df['id'].isin(unique_chembl_ids)]
print("size of filtered_molecule_df:", len(filtered_molecule_df))
filtered_molecule_table_mapped = pa.Table.from_pandas(filtered_molecule_df)

unique_chembl_ids_with_features = filtered_molecule_df['id'].unique()
unique_chembl_ids_with_names = filtered_molecule_df['name'].unique()

# Convert the filtered_molecule_df back to a PyArrow Table
filtered_molecule_table = pa.Table.from_pandas(filtered_molecule_df)
print("filtered_molecule_table:", filtered_molecule_table)
# save the filtered molecule table to a new CSV file
output_path = f'{results_path}filtered_molecule_table.csv'
filtered_molecule_df.to_csv(output_path, index=False)
# save as an xlsx file
output_path = f'{results_path}filtered_molecule_table.xlsx'
filtered_molecule_df.to_excel(output_path, index=False)
# # Print the counts to verify
# print("Original number of unique CHEMBL IDs:", len(unique_chembl_ids))
# print("Number of rows in filtered molecule table:", len(filtered_molecule_df))

# # print the exact different items between the original and filtered molecule table
# print("Items in the original molecule table but not in the filtered molecule table:")
# print(set(unique_chembl_ids) - set(filtered_molecule_df['id']))



drug_type = molecule_table.column('drugType').combine_chunks()



approved_drugs_list = list(unique_chembl_ids_with_features)
approved_drugs_list_name = list(unique_chembl_ids_with_names)
# save the approved drugs list to a csv file
approved_drugs_list_df = pd.DataFrame(approved_drugs_list, columns=['id'])



#extract the linked genes from the linkedTargets column
molecules_linked_genes_table = filtered_molecule_table.select(['id','linkedTargets.rows']).drop_null()
molecules_linked_genes = molecules_linked_genes_table.column('id').combine_chunks()


print("Number of molecules with linked genes:", len(molecules_linked_genes))

#extract the linked genes from the linkedTargets column
linked_genes = filtered_molecule_table.column('linkedTargets.rows').combine_chunks()

#Gene dataset
gene_dataset = ds.dataset(gene_path, format="parquet")
gene_table = gene_dataset.to_table().flatten().flatten()


#filter for genes linked to approved drugs
gene_filter_mask = pc.is_in(gene_table.column('id'), value_set= pc.unique(linked_genes.flatten()))
filtered_gene_table = gene_table.filter(gene_filter_mask)
# save it to a csv file
filtered_gene_table.to_pandas().to_csv(f"{results_path}filtered_gene_table.csv", index=False)

#cases for different training versions
if training_version == 21.04 or training_version == 21.06:
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','bioType', 'proteinAnnotations.functions', 'reactome']).flatten()
    gene_reactome_table = filtered_gene_table.select(['id', 'reactome']).flatten()

else: 
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','biotype', 'functionDescriptions', 'proteinIds', 'pathways'])
    filtered_gene_table = filtered_gene_table.select(['id', 'approvedName','biotype', 'functionDescriptions', 'proteinIds', 'pathways']).flatten()
    gene_reactome_table = filtered_gene_table.select(['id', 'pathways']).flatten().to_pandas()
    exploded = gene_reactome_table.explode('pathways')
    # Step 2: Extract the 'pathwayId' from the dictionaries in the 'pathways' column
    exploded['pathwayId'] = exploded['pathways'].apply(lambda x: x['pathwayId'] if pd.notnull(x) else None)
    # Step 3: Create a new DataFrame with just the 'id' and 'pathwayId' columns
    final_df = exploded[['id', 'pathwayId']]
    # Step 4: Convert the pandas DataFrame back to a PyArrow Table
    gene_reactome_table = pa.Table.from_pandas(final_df)
    gene_reactome_table = gene_reactome_table.drop_null()
    

if training_version == 21.04 or training_version == 21.06:
    proteinAnnotations = filtered_gene_table.column('proteinAnnotations.functions').combine_chunks()
else:
    proteinAnnotations = filtered_gene_table.column('functionDescriptions').combine_chunks()

#Gene nodes
gene = filtered_gene_table.column('id').combine_chunks()

if 'pathways' in filtered_gene_table.column_names:
    reactome = filtered_gene_table.column('pathways').combine_chunks().flatten()
    reactome = reactome.field(0)

else:
    reactome = filtered_gene_table.column('reactome').combine_chunks().flatten()


#Disease dataset
disease_dataset = ds.dataset(disease_path, format="parquet")
disease_table = disease_dataset.to_table()
# delete the row if the therapeuticAreas is empty
disease_table = disease_table.filter(pc.list_value_length(pc.field("therapeuticAreas")) > 0)
# Filter out rows where 'therapeuticAreas' contains 'EFO_0001444'

# Convert PyArrow Table to Pandas DataFrame
df = disease_table.to_pandas()


# Filter rows where 'therapeuticAreas' does NOT contain 'EFO_0001444'
filtered_df = df[~df['therapeuticAreas'].apply(lambda x: 'EFO_0001444' in x)]

# Convert back to PyArrow Table if needed
disease_table = pa.Table.from_pandas(filtered_df)
#disease_filter_mask = pc.is_in(disease_table.column('id'), value_set= pc.unique(approvedIndications.flatten()))
#filtered_disease_table = disease_table.filter(disease_filter_mask)
disease_table = disease_table.select(['id', 'name', 'description', 'ancestors', 'descendants', 'children', 'therapeuticAreas'])
disease_name = disease_table.column('name').combine_chunks()
description = disease_table.column('description').combine_chunks()



# Define the prefixes to filter out
prefixes_to_remove = ["UBERON", "ZFA", "CL", "GO", "FBbt", "FMA"]

# Create individual conditions for each prefix
filter_conditions = [
    pc.starts_with(disease_table.column('id'), prefix) for prefix in prefixes_to_remove
]

# Combine conditions iteratively using pc.or_
combined_filter = filter_conditions[0]
for condition in filter_conditions[1:]:
    combined_filter = pc.or_(combined_filter, condition)

# Negate the combined filter to keep rows not starting with the prefixes
negated_filter = pc.invert(combined_filter)

# Apply the filter condition to the table
filtered_disease_table = disease_table.filter(negated_filter)
# remove the rows with the descendants column is not empty
filtered_disease_table = filtered_disease_table.filter(pc.list_value_length(pc.field("descendants")) == 0)
# remove the rows with the id is EFO_0000544
filtered_disease_table = filtered_disease_table.filter(pc.field("id") != "EFO_0000544")


## manually map some annotations which are not well anotated

# Apply the updated redundant ID mapping
id_to_parentid_mapping_D = {k: resolve_mapping(v, redundant_id_mapping_D) for k, v in redundant_id_mapping_D.items()}
# Apply mapping to `approvedIndications` column (update disease IDs inside lists)
print("Before updating approvedIndications:")
print(filtered_indication_df[['id', 'approvedIndications']].head())
filtered_indication_df['approvedIndications'] = filtered_indication_df['approvedIndications'].apply(
    lambda x: update_approved_indications(x, id_to_parentid_mapping_D)
)

print(filtered_indication_df[['id', 'approvedIndications']].head())


# print("Unique CHEMBL IDs:")
# print(unique_chembl_ids)
# print("Number of unique CHEMBL IDs:", len(unique_chembl_ids))


# # Filter the molecule table to only keep rows where id is in unique_chembl_ids
# filtered_molecule_df = filtered_molecule_df[filtered_molecule_df['id'].isin(unique_chembl_ids)]

filtered_disease_table = filtered_disease_table.filter(pc.field("id") != "EFO_1000905")
filtered_disease_table = filtered_disease_table.filter(pc.field("id") != "EFO_0005752")
filtered_disease_table = filtered_disease_table.filter(pc.field("id") != "EFO_0007512")
filtered_disease_df = filtered_disease_table.to_pandas()
# save the updated filtered disease table to a new CSV file
output_path = f'{results_path}updated_filtered_disease_table.csv'
filtered_disease_df.to_csv(output_path, index=False)


unique_disease_ids = filtered_disease_df[filtered_disease_df.columns[0]].unique()

print("Number of unique disease IDs:", len(unique_disease_ids))

filtered_disease_df = filtered_disease_df[filtered_disease_df['id'].astype(str).isin(unique_disease_ids)]
print("Number of rows in filtered disease table:", len(filtered_disease_df))
filtered_disease_table_mapped = pa.Table.from_pandas(filtered_disease_df)

# Convert the filtered_molecule_df back to a PyArrow Table
filtered_disease_table = pa.Table.from_pandas(filtered_disease_df)

# Verify the filtering
print("Original disease table length:", len(disease_table))
print("Filtered disease table length:", len(filtered_disease_table))

pd.DataFrame(filtered_disease_table.to_pandas()).to_csv(dict_path + "/filtered_disease_table_new.csv")
# save as an xlsx file
pd.DataFrame(filtered_disease_table.to_pandas()).to_excel(dict_path + "/filtered_disease_table_new.xlsx")




#Approved disease nodes
disease = filtered_disease_table.column('id').combine_chunks()
disease_name = filtered_disease_table.column('name').combine_chunks()
#Include non approved diseases
if disease_similarity_network == True:
    disease_descendants = disease_table.column('descendants').combine_chunks().flatten()
    disease_children = disease_table.column('children').combine_chunks().flatten()
    disease_ancestors = disease_table.column('ancestors').combine_chunks().flatten()

    # Add descendants, children and ancestors to the disease similarity network
    df0 = disease.to_pandas()
    df1 = disease_descendants.unique().to_pandas()
    df2 = disease_children.unique().to_pandas()
    df3 = disease_ancestors.unique().to_pandas()

    #concatenate all the dataframes and story only unique values
    all_diseases_df = pd.concat([df0, df1, df2, df3], ignore_index=True).drop_duplicates()
    #convert to pyarrow array
    all_diseases = pa.array(all_diseases_df)
    logging.info("Disease similarity network nodes:", len(all_diseases_df))


# #Exclude non approved diseases
therapeutic_area = disease_table.column('therapeuticAreas').combine_chunks().flatten()


#Load associations dataset
# Options include 'associationByDatasourceDirect', 'associationByDatasourceIndirect', 'associationByDatatypeDirect', 'associationByDatatypeIndirect', 'associationByOverallDirect', 'associationByOverallIndirect'
associations_dataset = ds.dataset(associations_path, format="parquet")
associations_table = associations_dataset.to_table()

# Define Score column from associations table

for col in associations_table.column_names:
   if "Score" in col:
       logging.info(col)
       score_column = col

for col in associations_table.column_names:
   if "score" in col:
       logging.info(col)
       score_column = col


#Edge case for training version 21.04
if training_version == 21.04:
   associations_table = associations_table.select(['diseaseId', 'targetId', score_column])
else:
   associations_table = associations_table.select(['diseaseId', 'targetId', score_column])

#Filter for associations for genes linked with approved drugs
gene_filter_mask = pc.is_in(associations_table.column('targetId'), value_set= pc.unique(linked_genes.flatten()))
gene_filtered_associations_table = associations_table.filter(gene_filter_mask)

#Filter for associations for diseases with approved drugs 
disease_filter_mask = pc.is_in(gene_filtered_associations_table.column('diseaseId'), value_set= pc.unique(disease.unique()))
filtered_associations_table = gene_filtered_associations_table.filter(disease_filter_mask)

if training_version == 21.04:
   score_threshold = pc.field(score_column) >= 0.01
else:
   score_threshold = pc.field(score_column) >= 0.01

# Filter for associations with a score greater than or equal to threshold
filtered_associations_table = filtered_associations_table.filter(score_threshold)


drug_type_list = drug_type.drop_null().unique().to_pylist()

gene_list = gene.unique().to_pylist()
reactome_list = reactome.unique().to_pylist()
#disease_list from disease similarity network
if disease_similarity_network:
    disease_list = all_diseases.unique().to_pylist()
else:
    disease_list = disease.unique().to_pylist()
# length of disease list
print(len(disease_list))
print(".....")
disease_list_name = disease_name.unique().to_pylist()

therapeutic_area_list = therapeutic_area.unique().to_pylist()

node_info = {}

# Add node_info as key value pairs
node_info["Drugs"] = len(approved_drugs_list)
node_info["Drug Types"] = len(drug_type_list)
node_info["Genes"] = len(gene_list)
node_info["Reactome pathways"] = len(reactome_list)
node_info["Diseases"] = len(disease_list)
node_info["Therapeutic areas"] = len(therapeutic_area_list)
# print therapeutic area list 
print(therapeutic_area_list)

print(node_info)


drug_key_mapping = {approved_drugs_list[i]: i for i in range(len(approved_drugs_list))}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(drug_key_mapping, orient='index').to_csv(dict_path + "/drug_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(drug_key_mapping, orient='index').to_excel(dict_path + "/drug_key_mapping.xlsx")
drug_type_key_mapping = {drug_type_list[i]: i + len(drug_key_mapping) for i in range(len(drug_type_list))}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(drug_type_key_mapping, orient='index').to_csv(dict_path + "/drug_type_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(drug_type_key_mapping, orient='index').to_excel(dict_path + "/drug_type_key_mapping.xlsx")
gene_key_mapping = {gene_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) for i in range(len(gene_list))}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(gene_key_mapping, orient='index').to_csv(dict_path + "/gene_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(gene_key_mapping, orient='index').to_excel(dict_path + "/gene_key_mapping.xlsx")
reactome_key_mapping = {reactome_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) for i in range(len(reactome_list))}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(reactome_key_mapping, orient='index').to_csv(dict_path + "/reactome_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(reactome_key_mapping, orient='index').to_excel(dict_path + "/reactome_key_mapping.xlsx")
disease_key_mapping = {disease_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) + len(reactome_key_mapping) for i in range(len(disease_list))}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(disease_key_mapping, orient='index').to_csv(dict_path + "/disease_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(disease_key_mapping, orient='index').to_excel(dict_path + "/disease_key_mapping.xlsx")
therapeutic_area_key_mapping = {therapeutic_area_list[i]: i + len(drug_key_mapping) + len(drug_type_key_mapping) + len(gene_key_mapping) + len(reactome_key_mapping) + len(disease_key_mapping) for i in range(len(therapeutic_area_list) )}
# print as a dictionary in a csv file
pd.DataFrame.from_dict(therapeutic_area_key_mapping, orient='index').to_csv(dict_path + "/therapeutic_area_key_mapping.csv")
# print as a dictionary in a xlsx file
pd.DataFrame.from_dict(therapeutic_area_key_mapping, orient='index').to_excel(dict_path + "/therapeutic_area_key_mapping.xlsx")



#save all key_mappings in csvs at dict_path
# pd.DataFrame.from_dict(drug_key_mapping, orient='index').to_csv(f"{dict_path}/drug_key_mapping.csv")
# pd.DataFrame.from_dict(drug_type_key_mapping, orient='index').to_csv(f"{dict_path}/drug_type_key_mapping.csv")
# pd.DataFrame.from_dict(gene_key_mapping, orient='index').to_csv(f"{dict_path}/gene_key_mapping.csv")
# pd.DataFrame.from_dict(reactome_key_mapping, orient='index').to_csv(f"{dict_path}/reactome_key_mapping.csv")
# pd.DataFrame.from_dict(disease_key_mapping, orient='index').to_csv(f"{dict_path}/disease_key_mapping.csv")
# pd.DataFrame.from_dict(therapeutic_area_key_mapping, orient='index').to_csv(f"{dict_path}/therapeutic_area_key_mapping.csv")

a_list = ['associationByDatasourceDirect', 'associationByDatasourceIndirect', 'associationByDatatypeDirect', 'associationByDatatypeIndirect', 'associationByOverallDirect', 'associationByOverallIndirect']



#Start of feature block ---------------------------------
# Define node types
drug_node_type = 0
drug_type_node_type = 1
gene_node_type = 2
disease_node_type = 3
reactome_node_type = 4
therapeutic_area_node_type = 5
num_node_types = 6

# Get indices for each node_type
drug_indices = torch.tensor(get_indices_from_keys(approved_drugs_list, drug_key_mapping), dtype=torch.long)
drug_type_indices = torch.tensor(get_indices_from_keys(drug_type_list, drug_type_key_mapping), dtype=torch.long)
gene_indices = torch.tensor(get_indices_from_keys(gene_list, gene_key_mapping), dtype=torch.long)
reactome_indices = torch.tensor(get_indices_from_keys(reactome_list, reactome_key_mapping), dtype=torch.long)
disease_indices = torch.tensor(get_indices_from_keys(disease_list, disease_key_mapping), dtype=torch.long)
therapeutic_area_indices = torch.tensor(get_indices_from_keys(therapeutic_area_list, therapeutic_area_key_mapping), dtype=torch.long)
print(len(drug_indices), len(drug_type_indices), len(gene_indices), len(reactome_indices), len(disease_indices), len(therapeutic_area_indices))
#Feature extraction 
print("here")
print(disease_indices[0])
# Drug feature extraction

blackBoxWarning = filtered_molecule_table.column('blackBoxWarning').combine_chunks()
#print unique values in blackBoxWarning
print("blackBoxWarning:", blackBoxWarning.unique())

blackBoxWarning_vector = boolean_encode(blackBoxWarning, drug_indices)

#pad vector with -1 to make it the same length as drug_indices

# drug_name = filtered_molecule_table.column('name').combine_chunks()
# drug_name_vector = word_embeddings(drug_name)
yearOfFirstApproval = filtered_molecule_table.column('yearOfFirstApproval').combine_chunks()


yearOfFirstApproval_vector = normalize(yearOfFirstApproval, drug_indices)




hasBeenWithdrawn = filtered_molecule_table.column('hasBeenWithdrawn').combine_chunks()

hasBeenWithdrawn_vector = boolean_encode(hasBeenWithdrawn, drug_indices)


drug_one_hot = [1.0, 0.0 , 0.0, 0.0, 0.0, 0.0]
drug_node_type_vector = torch.tensor([drug_one_hot], dtype=torch.float32).repeat(len(drug_indices), 1)  # Resulting shape [length, 6]

#Concatenate the feature vectors along columns (dim=1)
#drug_feature_matrix = torch.cat((blackBoxWarning_vector, yearOfFirstApproval_vector, hasBeenWithdrawn_vector, drug_name_vector, drug_node_type_vector), dim=1)
# Print the sizes of the tensors
print(f"drug_node_type_vector size: {drug_node_type_vector.size()}")
print(f"blackBoxWarning_vector size: {blackBoxWarning_vector.size()}")
print(f"yearOfFirstApproval_vector size: {yearOfFirstApproval_vector.size()}")
drug_feature_matrix = torch.cat((drug_node_type_vector, blackBoxWarning_vector, yearOfFirstApproval_vector), dim=1)
print("drug_feature_matrix:", drug_feature_matrix)



# Gene feature extraction
gene_indices = torch.tensor(get_indices_from_keys(gene_list, gene_key_mapping), dtype=torch.long)
# gene_name = filtered_gene_table.column('approvedName').combine_chunks()
# gene_name_vector = word_embeddings(gene_name)
if training_version == 21.04 or training_version == 21.06 or training_version == 21.09:
    bioType = filtered_gene_table.column('bioType').combine_chunks()
else:
    bioType = filtered_gene_table.column('biotype').combine_chunks()

bioType_vector = cat_encode(bioType, gene_indices).unsqueeze(1)


gene_one_hot = [0.0, 0.0 , 1.0, 0.0, 0.0, 0.0]


gene_node_type_vector = torch.tensor([gene_one_hot], dtype=torch.float32).repeat(len(gene_indices), 1)  # Resulting shape [length, 6]
#gene_feature_matrix = torch.cat((bioType_vector, gene_name_vector, gene_node_type_vector), dim=1)
gene_feature_matrix = torch.cat((gene_node_type_vector, bioType_vector), dim=1)
print("gene_feature_matrix:", gene_feature_matrix)
# show different values in the last column of gene_feature_matrix
print(gene_feature_matrix[:, -1].unique())



# Disease feature extraction
disease_indices = torch.tensor(get_indices_from_keys(disease_list, disease_key_mapping), dtype=torch.long)
#disease_name_vector = word_embeddings(disease_name)
disease_one_hot = [0.0, 0.0 , 0.0, 1.0, 0.0, 0.0]
disease_node_type_vector = torch.tensor([disease_one_hot], dtype=torch.float32).repeat(len(disease_indices), 1)  # Resulting shape [length, 6]
#disease_feature_matrix = torch.cat((disease_name_vector, disease_node_type_vector), dim=1)
disease_feature_matrix = disease_node_type_vector
print("disease_feature_matrix:", disease_feature_matrix)

#Nodes without features also need a feature matrix
# drugType feature extraction
drug_type_one_hot = [0.0, 1.0 , 0.0, 0.0, 0.0, 0.0]
drug_type_feature_matrix = torch.tensor([drug_type_one_hot], dtype=torch.float32).repeat(len(drug_type_indices), 1)  # Resulting shape [length, 6]

# reactome feature extraction
reactome_one_hot = [0.0, 0.0 , 0.0, 0.0, 1.0, 0.0]
reactome_feature_matrix = torch.tensor([reactome_one_hot], dtype=torch.float32).repeat(len(reactome_indices), 1)  # Resulting shape [length, 6]

# therapeutic_area feature extraction
therapeutic_area_one_hot = [0.0, 0.0 , 0.0, 0.0, 0.0, 1.0]
therapeutic_area_feature_matrix = torch.tensor([therapeutic_area_one_hot], dtype=torch.float32).repeat(len(therapeutic_area_indices), 1)  # Resulting shape [length, 6]

# Feature list
feature_map = {}
feature_map["Drug Features"] = ['blackBoxWarning', 'yearOfFirstApproval', 'node_type']
feature_map["Drug type Features"] = ['node_type']
feature_map["Disease Features"] = ['node_type']
feature_map["Gene Features"] = ['node_type', 'bioType']
feature_map["Reactome Features"] = ['node_type']
feature_map["Therapeutic area Features"] = ['node_type']     

pad_size = 9
# Pad feature matrices



drug_feature_matrix = pad_feature_matrix(drug_feature_matrix, pad_size, -1)
drug_type_feature_matrix = pad_feature_matrix(drug_type_feature_matrix, pad_size, -1)
gene_feature_matrix = pad_feature_matrix(gene_feature_matrix, pad_size, -1)
disease_feature_matrix = pad_feature_matrix(disease_feature_matrix, pad_size, -1)
reactome_feature_matrix = pad_feature_matrix(reactome_feature_matrix, pad_size, -1)
therapeutic_area_feature_matrix = pad_feature_matrix(therapeutic_area_feature_matrix, pad_size, -1)


global_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot', 'blackBoxWarning', 'yearOfFirstApproval', 'bioType']
def align_features(matrix, feature_columns, global_feature_columns):
    aligned_matrix = torch.zeros(matrix.size(0), len(global_feature_columns)) - 1  
    for idx, col in enumerate(feature_columns):
        global_idx = global_feature_columns.index(col)
        aligned_matrix[:, global_idx] = matrix[:, idx]
    return aligned_matrix
drug_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot', 'blackBoxWarning', 'yearOfFirstApproval']
drug_type_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot']
gene_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot', 'bioType']
disease_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot']
reactome_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot']
therapeutic_area_feature_columns = ['drug_one_hot', 'drug_type_one_hot', 'gene_one_hot', 'disease_one_hot', 'reactome_one_hot', 'therapeutic_area_one_hot']



aligned_drug_matrix = align_features(drug_feature_matrix, drug_feature_columns, global_feature_columns)
print("Aligned drug matrix:", aligned_drug_matrix)
print("Aligned drug matrix shape:", aligned_drug_matrix.shape)
aligned_drug_type_matrix = align_features(drug_type_feature_matrix, drug_type_feature_columns, global_feature_columns)
print("Aligned drug type matrix:", aligned_drug_type_matrix)
print("Aligned drug type matrix shape:", aligned_drug_type_matrix.shape)

print(gene_feature_matrix[:, 6].unique())
aligned_gene_matrix = align_features(gene_feature_matrix, gene_feature_columns, global_feature_columns)
print(aligned_gene_matrix[:, 6].unique())

print("Aligned gene matrix:", aligned_gene_matrix)
print("Aligned gene matrix shape:", aligned_gene_matrix.shape)
aligned_disease_matrix = align_features(disease_feature_matrix, disease_feature_columns, global_feature_columns)
print("Aligned disease matrix:", aligned_disease_matrix)
print("Aligned disease matrix shape:", aligned_disease_matrix.shape)
aligned_reactome_matrix = align_features(reactome_feature_matrix, reactome_feature_columns, global_feature_columns)
print("Aligned reactome matrix:", aligned_reactome_matrix)
print("Aligned reactome matrix shape:", aligned_reactome_matrix.shape)
aligned_therapeutic_area_matrix = align_features(therapeutic_area_feature_matrix, therapeutic_area_feature_columns, global_feature_columns)
print("Aligned therapeutic area matrix:", aligned_therapeutic_area_matrix)
print("Aligned therapeutic area matrix shape:", aligned_therapeutic_area_matrix.shape)
# print the unique values in the last column of aligned_gene_matrix
print(aligned_drug_matrix[:, 6].unique())
print(aligned_gene_matrix[:, 8].unique())
# Assuming feature matrices can be stacked without compatibility issues
all_features = torch.vstack([aligned_drug_matrix, aligned_drug_type_matrix, aligned_gene_matrix, aligned_disease_matrix, aligned_reactome_matrix, aligned_therapeutic_area_matrix]) 
print(all_features[:, 6].unique())
print(all_features[:, 8].unique())

print("Size of feature matrix: ", all_features.shape)
print("Feature matrix:", all_features)
#End of feature block ---------------------------------



# Edge extraction

# Default Edge tables
molecule_drugType_table = filtered_molecule_table_mapped.select(['id', 'drugType']).drop_null().flatten()
molecule_disease_table = filtered_indication_table_mapped.select(['id', 'approvedIndications']).flatten()
molecule_gene_table = filtered_molecule_table_mapped.select(['id', 'linkedTargets.rows']).drop_null().flatten()
gene_reactome_table = gene_reactome_table #extracted according to version above
disease_therapeutic_table = filtered_disease_table_mapped.select(['id', 'therapeuticAreas']).drop_null().flatten()
disease_gene_table = filtered_associations_table.select(['diseaseId', 'targetId']).flatten()

print("@@@@")
print(molecule_disease_table)
# print pyarrow.Table to pandas dataframe
print(molecule_disease_table.to_pandas())
# save as csv
pd.DataFrame(molecule_disease_table.to_pandas()).to_csv(dict_path + "/molecule_disease_table.csv")


# Extract edges as tensors
molecule_drugType_edges = extract_edges(molecule_drugType_table, drug_key_mapping, drug_type_key_mapping)
# avoid duplicates
molecule_drugType_edges = torch.unique(molecule_drugType_edges, dim=1)
molecule_disease_edges = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping)
# avoid duplicates
molecule_disease_edges = torch.unique(molecule_disease_edges, dim=1)
molecule_gene_edges = extract_edges(molecule_gene_table, drug_key_mapping, gene_key_mapping)
# avoid duplicates
molecule_gene_edges = torch.unique(molecule_gene_edges, dim=1)

gene_reactome_edges = extract_edges(gene_reactome_table, gene_key_mapping, reactome_key_mapping) 
# avoid duplicates
gene_reactome_edges = torch.unique(gene_reactome_edges, dim=1)
disease_therapeutic_edges = extract_edges(disease_therapeutic_table, disease_key_mapping, therapeutic_area_key_mapping)
# avoid duplicates
disease_therapeutic_edges = torch.unique(disease_therapeutic_edges, dim=1)

disease_gene_edges = extract_edges(disease_gene_table, disease_key_mapping, gene_key_mapping)
# avoid duplicates
disease_gene_edges = torch.unique(disease_gene_edges, dim=1)

test = extract_edges_no_mapping(molecule_disease_table)
#print("Test:", test)
# save test as csv
pd.DataFrame(test).to_csv(dict_path + "/test.csv")


print("Molecule to disease edges:", molecule_disease_edges)
#Extract edges as lists
molecule_drugType_edge_list = extract_edges(molecule_drugType_table, drug_key_mapping, drug_type_key_mapping, return_edge_list=True)
molecule_disease_edge_list = extract_edges(molecule_disease_table, drug_key_mapping, disease_key_mapping, return_edge_list=True)
molecule_gene_edge_list = extract_edges(molecule_gene_table, drug_key_mapping, gene_key_mapping, return_edge_list=True)
gene_reactome_edge_list = extract_edges(gene_reactome_table, gene_key_mapping, reactome_key_mapping, return_edge_list=True)
disease_therapeutic_edge_list = extract_edges(disease_therapeutic_table, disease_key_mapping, therapeutic_area_key_mapping, return_edge_list=True)
disease_gene_edge_list = extract_edges(disease_gene_table, disease_key_mapping, gene_key_mapping, return_edge_list=True)

edge_info = {}
edge_info["Molecule to Drug Type edges"] = len(molecule_drugType_edges[0])
edge_info["Molecule to disease edges"] = len(molecule_disease_edges[0])
edge_info["Molecule to gene edges"] = len(molecule_gene_edges[0])
edge_info["Gene to reactome edges"] = len(gene_reactome_edges[0])
edge_info["Disease to therapeutic edges"] = len(disease_therapeutic_edges[0])
edge_info["Disease to gene edges"] = len(disease_gene_edges[0])

print(edge_info)
existing_drug_disease_edges = list(zip(molecule_disease_edges[0].tolist(), molecule_disease_edges[1].tolist()))
existing_drug_disease_edges_array = np.array(existing_drug_disease_edges)
print("Unique drug nodes:", len(set(existing_drug_disease_edges_array[:, 0].tolist())))
print("Unique disease nodes:", len(set(existing_drug_disease_edges_array[:, 1].tolist())))


all_edge_index = torch.cat([molecule_drugType_edges, molecule_disease_edges, molecule_gene_edges, gene_reactome_edges, 
                                disease_therapeutic_edges, disease_gene_edges], dim=1)


def randomize_tensor(tensor):
    # Randomize the tensor
    tensor = tensor[:, torch.randperm(tensor.size(1))]
    return tensor


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers

        # Initial GCNConv layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Additional GCNConv layers
        self.conv_list = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GCNConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional GCNConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x

class TransformerNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(TransformerNet, self).__init__()
        self.num_layers = num_layers

        # Initial TransformerConv layer with concat=False
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)

        # Additional TransformerConv layers
        self.conv_list = torch.nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First TransformerConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional TransformerConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(SAGENet, self).__init__()
        self.num_layers = num_layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv_list = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:
                x = F.relu(x)
                x = self.dropout(x)
        x = self.final_layer(x)
        return x




  
all_molecule_disease = generate_pairs(approved_drugs_list, disease_list, drug_key_mapping, disease_key_mapping)
print("All molecule disease pairs:", len(all_molecule_disease))

# simpathic_disease_list = ['Orphanet_98757', 'MONDO_0100186', 'Orphanet_912']
# simpathic_molecule_disease = generate_pairs(approved_drugs_list, simpathic_disease_list, drug_key_mapping, disease_key_mapping)
# print("Simpathic molecule disease pairs:", len(simpathic_molecule_disease))


existing_drug_disease_edges = list(zip(molecule_disease_edges[0].tolist(), molecule_disease_edges[1].tolist()))
existing_drug_disease_edges_set = set(existing_drug_disease_edges)
print("Existing drug disease edges:", len(existing_drug_disease_edges))
# print 100 first elements of the existing_drug_disease_edges
print("First 100 elements of the existing_drug_disease_edges:", existing_drug_disease_edges[:100])
# map them back to drug_key_mapping and disease_key_mapping

existing_drug_disease_edges_str = [(approved_drugs_list_name[i], disease_list_name[j-3644]) for i, j in existing_drug_disease_edges]
# save the str into a csv with the column names
pd.DataFrame(existing_drug_disease_edges_str, columns = ['drug_name', 'disease_name']).to_csv(dict_path + "/training_drug_disease_edges_name.csv")
training_drug_disease_edges_name = pd.DataFrame(existing_drug_disease_edges_str, columns = ['drug_name', 'disease_name'])
# save as an xlsx file
pd.DataFrame(existing_drug_disease_edges_str, columns = ['drug_name', 'disease_name']).to_excel(dict_path + "/training_drug_disease_edges_name.xlsx")

# Convert the positive edge index to a tensor
pos_edge_index = torch.tensor(existing_drug_disease_edges).T
print("Positive edge index:", pos_edge_index)
print("Positive edge index shape:", pos_edge_index.shape)

train_md_edges_set, new_val_edges_set = generate_edge_sets_val(training_version, validation_version)
new_test_edges_set = generate_edge_sets_test(training_version, validation_version, test_version, new_val_edges_set)
not_linked_set = list(set(all_molecule_disease) - train_md_edges_set)    
not_linked_list = list(not_linked_set)

# 1. not_linked_set_train
random.shuffle(not_linked_list)
not_linked_set_train = set(not_linked_list)

# 2. not_linked_set_val
not_linked_set_val = list(set(all_molecule_disease) - train_md_edges_set- new_val_edges_set)
random.shuffle(not_linked_set_val)

# 3. not_linked_set_test
not_linked_set_test = list(set(all_molecule_disease) - train_md_edges_set- new_val_edges_set - new_test_edges_set)
random.shuffle(not_linked_set_test)

'''
# Step 3: Split into training, validation, and test sets
split_index_train = int(len(not_linked_list) * 0.8)
split_index_val = int(len(not_linked_list) * 0.9)

not_linked_set_train = set(not_linked_list[:split_index_train])
not_linked_set_val = set(not_linked_list[split_index_train:split_index_val])
not_linked_set_test = set(not_linked_list[split_index_val:])
# combine simpathic pairs with the not_linked_set_test


# print first 5 elements of the not_linked_set_train
print("First 5 elements of the not_linked_set_train:", list(not_linked_set_train)[:5])

# Step 4: Sanity checks to ensure no overlap
assert len(not_linked_set_train.intersection(not_linked_set_val)) == 0, "Overlap detected between train and val sets!"
assert len(not_linked_set_train.intersection(not_linked_set_test)) == 0, "Overlap detected between train and test sets!"
assert len(not_linked_set_val.intersection(not_linked_set_test)) == 0, "Overlap detected between val and test sets!"




# Print the sizes for verification
print(f"Training set (not linked): {len(not_linked_set_train)}")
print(f"Validation set (not linked): {len(not_linked_set_val)}")
print(f"Test set (not linked): {len(not_linked_set_test)}")
'''
val_edge_tensor, label_tensor, false_pairs_val = create_validation_tensors(new_val_edges_set, not_linked_set_val)

# check if there is any overlap between the validation and test sets

assert len(new_val_edges_set.intersection(train_md_edges_set)) == 0, "Overlap detected between val and train sets!"
#check if there is any overlap between the test and training sets




datetime = dt.datetime.now().strftime("%Y%m%d%H%M%S")
metadata = {"node_info" : node_info, "feature_map": feature_map, "edge_info" : edge_info}

# Create the homogenous graph object
graph = Data(x=all_features, edge_index=all_edge_index, val_edge_index=val_edge_tensor, val_edge_label=label_tensor, metadata=metadata)
graph = T.ToUndirected()(graph)  # Convert to undirected graph
print("Graph Validated:", graph.validate())
print(graph)
# save the graph object to a file

torch.save(graph, f'{results_path}{training_version}_{negative_sampling_approach}_{as_dataset}_{datetime}_graph.pt')
print("Graph saved")
graph.x = graph.x.float()  # Ensure node features are float32
graph.edge_index = graph.edge_index.long()  # This is usually already the case and is fine as is

num_neg_samples = len(existing_drug_disease_edges)
fixed_neg_edge_index = torch.tensor(
    random.sample(list(not_linked_set_train), num_neg_samples), dtype=torch.long
).T  # Ensure (2, N) format

neg_edge_index = fixed_neg_edge_index

test_edge_tensor, test_label_tensor = create_test_tensors(new_test_edges_set, not_linked_set_test) 


# Clear GPU memory
torch.cuda.empty_cache()

# Define the device to be loaded
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = graph.to(device)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_test_all(models_to_train, graph, pos_edge_index, neg_edge_index, val_edge_tensor, label_tensor, test_edge_tensor, test_label_tensor, results_path, datetime, num_epochs=1000, patience=10):
    """Train and test multiple models."""
    all_results = []
    trained_models = {}  # Dictionary to store models

    for model_name, model_class in models_to_train.items():
        torch.cuda.empty_cache()
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        print(f"\nTraining {model_name}")

        # Create the model with the appropriate architecture
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=16,
            out_channels=16,
            dropout_rate=0.5
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        best_threshold, best_model_path = train_one_model(
            model, optimizer, graph, pos_edge_index, neg_edge_index,
            val_edge_tensor, label_tensor, num_epochs, patience,
            results_path, model_name
        )

        # Load the best-trained model before testing
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        test_probs, test_preds, metrics = test_model(
            model, best_model_path, graph, test_edge_tensor,
            test_label_tensor, best_threshold
        )

        print(f"Test metrics for {model_name}: {metrics}")

        # Create and save confusion matrix
        cm = confusion_matrix(test_label_tensor.cpu().numpy(), test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'], rotation=0)
        plt.tight_layout()
        plt.savefig(f'{results_path}{model_name}_confusion_matrix_{datetime}.png')
        plt.close()

        # Print confusion matrix values
        print(f"\nConfusion Matrix for {model_name}:")
        print(f"True Positives: {cm[1][1]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"True Negatives: {cm[0][0]}")
        print(f"False Negatives: {cm[1][0]}")
        
        # Calculate additional metrics from confusion matrix
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        ppv = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        npv = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
        
        print(f"Sensitivity/Recall: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision/PPV: {ppv:.4f}")
        print(f"NPV: {npv:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"APR: {metrics['apr']:.4f}")

        all_results.append({
            'name': model_name,
            'labels': test_label_tensor.cpu().numpy(),
            'probs': test_probs,
            'confusion_matrix': cm,
            'metrics': {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'apr': metrics['apr']
            }
        })

        # Store trained model for return
        trained_models[model_name] = model

    # After all models, save a summary of results
    summary_file = f'{results_path}model_comparison_summary_{datetime}.txt'
    with open(summary_file, 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("=======================\n\n")
        
        for res in all_results:
            model = res['name']
            metrics = res['metrics']
            cm = res['confusion_matrix']
            
            f.write(f"Model: {model}\n")
            f.write("-----------------\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"  TN: {cm[0][0]}, FP: {cm[0][1]}\n")
            f.write(f"  FN: {cm[1][0]}, TP: {cm[1][1]}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Sensitivity/Recall: {metrics['sensitivity']:.4f}\n")
            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"  Precision/PPV: {metrics['ppv']:.4f}\n")
            f.write(f"  NPV: {metrics['npv']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  AUC: {metrics['auc']:.4f}\n")
            f.write(f"  APR: {metrics['apr']:.4f}\n\n")
        
        # Add overall comparison
        f.write("Model Ranking by AUC:\n")
        for idx, res in enumerate(sorted(all_results, key=lambda x: x['metrics']['auc'], reverse=True)):
            f.write(f"{idx+1}. {res['name']}: {res['metrics']['auc']:.4f}\n")
        
        f.write("\nModel Ranking by APR:\n")
        for idx, res in enumerate(sorted(all_results, key=lambda x: x['metrics']['apr'], reverse=True)):
            f.write(f"{idx+1}. {res['name']}: {res['metrics']['apr']:.4f}\n")
    
    print(f"\nDetailed summary saved to {summary_file}")

    # Plot comparative charts
    plot_multiple_roc_curves(all_results, results_path, datetime)
    plot_multiple_pr_curves(all_results, results_path, datetime)
    
    # Create comparative bar charts for key metrics
    plot_metric_comparison(all_results, 'auc', 'AUC Score Comparison', results_path, datetime)
    plot_metric_comparison(all_results, 'apr', 'Average Precision Score Comparison', results_path, datetime)
    plot_metric_comparison(all_results, 'f1', 'F1 Score Comparison', results_path, datetime)
    plot_metric_comparison(all_results, 'sensitivity', 'Sensitivity Comparison', results_path, datetime)
    plot_metric_comparison(all_results, 'specificity', 'Specificity Comparison', results_path, datetime)

    return trained_models

def plot_metric_comparison(all_results, metric_name, title, results_path, datetime):
    """Creates a bar chart comparing a specific metric across all models."""
    model_names = [res['name'] for res in all_results]
    metric_values = [res['metrics'][metric_name] for res in all_results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add the exact values on top of each bar
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.ylim(0, max(metric_values) + 0.1)  # Add some space above the highest bar
    plt.xlabel('Model')
    plt.ylabel(f'{metric_name.upper()} Score')
    plt.title(title)
    plt.xticks(rotation=0)  # Keep model names horizontal
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{results_path}{metric_name}_comparison_{datetime}.png')
    plt.close()

def plot_multiple_roc_curves(model_results, results_path, datetime):
    """Plots and saves ROC curves for multiple models with different line styles and increased legend size."""
    plt.figure(figsize=(10, 10))
    
    # Define custom colors
    colors = ['blue', 'red', 'orange']
    
    # Define different line styles
    line_styles = [':', '-.', '-']
    
    for i, res in enumerate(model_results):
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        roc_auc = auc(fpr, tpr)
        
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        plt.plot(fpr, tpr, 
                 linestyle=line_style,
                 color=color, 
                 lw=2, 
                 label=f"{res['name']} (AUC = {roc_auc:.4f})")

    # Reference line (diagonal)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.5, alpha=0.7)
    
    # Increased font sizes
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
    
    # Increased legend font size significantly
    plt.legend(loc='lower right', fontsize=18)
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(f'{results_path}comparison_roc_curves_{datetime}.png', dpi=300)
    plt.close()

def plot_multiple_pr_curves(model_results, results_path, datetime):
    """Plots and saves Precision-Recall curves for multiple models with optimized layout."""
    plt.figure(figsize=(10, 10))
    
    # Define custom colors
    colors = ['blue', 'red', 'orange']
    
    # Define different line styles
    line_styles = [':', '-.', '-']
    
    # Calculate baseline data first to determine appropriate axis limits
    min_precision = 1.0
    for res in model_results:
        precision, _, _ = precision_recall_curve(res['labels'], res['probs'])
        min_precision = min(min_precision, min(precision))
    
    # Round down to nearest 0.1 for y-axis minimum, but never below 0
    y_min = max(0, np.floor(min_precision * 10) / 10)
    
    for i, res in enumerate(model_results):
        precision, recall, _ = precision_recall_curve(res['labels'], res['probs'])
        avg_precision = average_precision_score(res['labels'], res['probs'])
        
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        plt.plot(recall, precision, 
                 linestyle=line_style,
                 color=color, 
                 lw=2, 
                 label=f"{res['name']} (AP = {avg_precision:.4f})")

    # Reference line for random classifier
    if model_results:
        prevalence = np.mean(model_results[0]['labels'])
        plt.plot([0, 1], [prevalence, prevalence], 
                 linestyle='--', color='gray', lw=1.5, alpha=0.7)
    
    # Increased font sizes
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
    
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([y_min, 1.05])
    
    # Increased legend font size significantly
    plt.legend(loc='best', fontsize=18)
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(f'{results_path}comparison_pr_curves_{datetime}.png', dpi=300)
    plt.close()
    
# Define the GCNConv model class
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers

        # Initial GCNConv layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Additional GCNConv layers
        self.conv_list = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First GCNConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional GCNConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x

# Define the TransformerConv model class
class TransformerModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers

        # Initial TransformerConv layer with concat=False
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=4, concat=False)

        # Additional TransformerConv layers
        self.conv_list = torch.nn.ModuleList(
            [TransformerConv(hidden_channels, hidden_channels, heads=4, concat=False) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First TransformerConv layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional TransformerConv layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x

# Define the SAGEConv model class
class SAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_rate=0.5):
        super(SAGEModel, self).__init__()
        self.num_layers = num_layers

        # Initial GraphSAGE layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # Additional hidden layers
        self.conv_list = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

        # Layer normalization and dropout
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Final output layer
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Additional layers
        for k in range(self.num_layers - 1):
            x = self.conv_list[k](x, edge_index)
            x = self.ln(x)
            if k < self.num_layers - 2:  # Apply activation and dropout except on the last hidden layer
                x = F.relu(x)
                x = self.dropout(x)

        # Final layer to produce output
        x = self.final_layer(x)
        return x

# Function to train a single model
def train_one_model(model, optimizer, graph, pos_edge_index, neg_edge_index, val_edge_tensor, label_tensor, num_epochs, patience, results_path, model_name):
    """Train a single model and return the best threshold and model path."""
    best_val_loss = float('inf')
    counter = 0
    best_threshold = 0.5
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_model_path = f'{results_path}{model_name}_best_model.pt'
    
    for epoch in tqdm(range(num_epochs), desc=f'Training {model_name}'):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z = model(graph.x.float(), graph.edge_index)
        
        # Compute scores
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        
        # Compute loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)
            val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
            val_loss = loss_function(val_scores, label_tensor.float())
            val_probs = torch.sigmoid(val_scores)
            val_threshold = val_probs.mean().item()
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_threshold = val_threshold
                counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}, Best threshold: {best_threshold:.4f}")
            else:
                counter += 1
                print(f"Epoch {epoch+1}: No improvement. Counter: {counter}")
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    return best_threshold, best_model_path

# Function to test a model
def test_model(model, model_path, graph, test_edge_tensor, test_label_tensor, threshold):
    """Test a model and return probabilities, predictions and metrics."""
    batch_size = 1000
    test_probs = []
    
    model.eval()
    with torch.no_grad():
        z = model(graph.x.float(), graph.edge_index)
        
        # Process test edges in batches
        for start in range(0, len(test_edge_tensor), batch_size):
            end = min(start + batch_size, len(test_edge_tensor))
            batch_test_edges = test_edge_tensor[start:end]
            
            # Calculate scores and probabilities
            batch_test_scores = (z[batch_test_edges[:, 0]] * z[batch_test_edges[:, 1]]).sum(dim=-1)
            batch_test_probs = torch.sigmoid(batch_test_scores)
            
            # Append probabilities
            test_probs.append(batch_test_probs.cpu().numpy())
    
    # Combine results
    test_probs = np.concatenate(test_probs)
    test_preds = (test_probs >= threshold).astype(float)
    
    # Calculate metrics
    TP = sum(test_preds * test_label_tensor.numpy())
    FP = sum(test_preds * (1 - test_label_tensor.numpy()))
    FN = sum((1 - test_preds) * test_label_tensor.numpy())
    TN = sum((1 - test_preds) * (1 - test_label_tensor.numpy()))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    apr_score = average_precision_score(test_label_tensor.cpu().numpy(), test_probs)
    auc_score = roc_auc_score(test_label_tensor.cpu().numpy(), test_probs)
    
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'apr': apr_score,
        'auc': auc_score
    }
    
    return test_probs, test_preds, metrics

# Now call the function with all three model types
trained_models = train_and_test_all(
    {
        'GCNModel': GCNModel,
        'SAGEModel': SAGEModel,
        'TransformerModel': TransformerModel
    },
    graph, pos_edge_index, neg_edge_index,
    val_edge_tensor, label_tensor,
    test_edge_tensor, test_label_tensor,
    results_path, datetime,
    num_epochs=1000, patience=10
)

# You can access individual models from the returned dictionary
gcn_model = trained_models['GCNModel']
transformer_model = trained_models['TransformerModel']
sage_model = trained_models['SAGEModel']