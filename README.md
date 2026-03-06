# KG-Bench: Benchmarking Graph Neural Networks for Drug Repurposing

This repository accompanies the paper "KG-Bench: Benchmarking Graph Neural Network Algorithms for Drug Repurposing". It provides a FAIR-compliant benchmarking framework for evaluating GNN architectures on drug–disease association prediction tasks, using knowledge graphs constructed from Open Targets data.

Drug repurposing, finding new therapeutic uses for existing drugs, is a promising strategy to accelerate drug development. KG-Bench addresses key challenges including:

+ Lack of standardized benchmarks
+ Data leakage between training and test sets
+ Imbalanced learning scenarios due to sparse negative samples

The framework uses time-stamped Open Targets releases (21.06 / 23.06 / 24.06) to create realistic temporal train / validation / test splits, enabling retrospective validation of model generalisation to newly reported drug–disease associations.

## This GitHub repository provides:

+ Scripts to construct biomedical knowledge graphs from Open Targets data
+ Preprocessed datasets for training, validation, and testing
+ Implementations of GNN models: GCNConv, GraphSAGE, and TransformerConv
+ Benchmarking pipeline with ablation studies and negative sampling strategies
+ Evaluation metrics including AUC, precision-recall curves, and more

## Project Structure

```
drug_disease_prediction/
├── src/                              # Shared modules
│   ├── __init__.py
│   ├── models.py                     # GNN model definitions
│   ├── utils.py                      # Evaluation metrics & helpers
│   ├── config.py                     # Configuration management
│   └── data_processing.py            # Data loading & preprocessing
│
├── scripts/                          # Main pipeline scripts
│   ├── 1_create_graph.py             # Knowledge graph construction
│   ├── 2_train_models.py             # Model training & validation
│   ├── 3_test_evaluate.py            # Model testing & evaluation
│   └── 4_explainer.py                # GNNExplainer attribution & visualizer
│
├── processed_data/                   # Pre-processed data files
├── run_pipeline.py                   # Pipeline orchestrator
├── requirements.txt
├── config.json                       # Configuration file
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drug_disease_prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

** Most users should choose Option 1. Only choose Option 2 if you need to work with raw Open Targets data or want to understand the full data processing pipeline.

---

### Option 1 — Use Pre-processed Data (Recommended)
**Best for:** Getting started quickly, running experiments, most research use cases

The repository includes pre-processed data files ready for immediate use. No additional downloads required.


**Expected directory structure:**
```
processed_data/
├── tables/
│   ├── processed_molecules.csv
│   ├── processed_indications.csv
│   ├── processed_diseases.csv
│   ├── processed_genes.csv
│   └── processed_associations.csv
├── mappings/
│   ├── drug_key_mapping.json
│   ├── drug_type_key_mapping.json
│   ├── gene_key_mapping.json
│   ├── reactome_key_mapping.json
│   ├── disease_key_mapping.json
│   ├── therapeutic_area_key_mapping.json
│   └── mapping_summary.json
└── edges/
    ├── 1_molecule_drugType_edges.pt
    ├── 2_molecule_disease_edges.pt
    ├── 3_molecule_gene_edges.pt
    ├── 4_gene_reactome_edges.pt
    ├── 5_disease_therapeutic_edges.pt
    ├── 6_disease_gene_edges.pt
    └── edge_statistics.json
```


---

### Option 2: Download Raw OpenTargets Data (Advanced)

**Best for:** Custom data processing, understanding the full pipeline, working with different OpenTargets versions

Requires ~50 GB free disk space. Visit https://platform.opentargets.org/downloads/ and download the following datasets in Parquet format:

#### Access the Data

Visit the OpenTargets downloads page: https://platform.opentargets.org/downloads/

#### Method A: Using FileZilla (Recommended)
1. **Host**: `ftp.ebi.ac.uk`
2. **Remote site**: `/pub/databases/opentargets/platform/`
3. **Navigate** to the version folders: `21.06`, `23.06`, or `24.06`
4. **Go to**: `output/etl/parquet/` within each version
5. **Download** the required datasets from each version

#### Method B: Command Line Download
```bash
# Create directory structure
mkdir -p raw_data/{21.06,23.06,24.06}

# Download using wget (example for 21.06)
cd raw_data/21.06
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/indication/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/molecule/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/disease/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/target/
wget -r -np -nH --cut-dirs=7 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/etl/parquet/associationByOverallDirect/

# Repeat for versions 23.06 and 24.06 (only indication needed for these)
```

#### Required Data by Version

**Training Version (21.06):**
From `/pub/databases/opentargets/platform/21.06/output/etl/parquet/`:
- `indication/`
- `molecule/`
- `disease/` → rename to `diseases/`
- `target/` → rename to `targets/`
- `associationByOverallDirect/`

**Validation Version (23.06):**
From `/pub/databases/opentargets/platform/23.06/output/etl/parquet/`:
- `indication/`

**Test Version (24.06):**
From `/pub/databases/opentargets/platform/24.06/output/etl/parquet/`:
- `indication/`

#### Final Directory Structure:
```
raw_data/
├── 21.06/
│   ├── indication/           
│   ├── molecule/            
│   ├── diseases/            # renamed from disease
│   ├── targets/             # renamed from target
│   └── associationByOverallDirect/
├── 23.06/
│   └── indication/          
└── 24.06/
    └── indication/          
```

**Important Notes:**
- All files are in PARQUET format
- The actual FTP path includes `/output/etl/parquet/` before the dataset names
- Rename `disease` to `diseases` and `target` to `targets` after download
- Large datasets may require significant download time and storage space
- Check OpenTargets license terms before using the data

After downloading, you'll need to update your `config.json` to point to the raw data directory and run the full processing pipeline.

---

## Usage

### Complete Pipeline
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Step 1: Build the knowledge graph
python scripts/1_create_graph.py --output-dir results/

# Step 2: Train GNN models
python scripts/2_train_models.py results/graph_*.pt --results-path results/models/

# Step 3: Evaluate models on the test set
python scripts/3_test_evaluate.py results/graph_*.pt results/models/ \
    --results-path results/ --export-fp

# Step 4: GNNExplainer attribution & interactive visualizer
python scripts/4_explainer.py results/graph_*.pt results/explainer/ \
    --summary results/models/training_summary_*.json \
    --fp-csv  results/predictions/*TransformerModel*_FP_*.csv \
    --model   TransformerModel
```

## Configuration

Create a `config.json` file:

```json
{
  "training_version": 21.06,
  "validation_version": 23.06,
  "test_version": 24.06,
  "as_dataset": "associationByOverallDirect",
  "negative_sampling_approach": "random",
  "processed_path": "processed_data/"
}
```
## Models Supported

| Architecture | Class name | 
|---|---|
| Graph Convolutional Network | `GCNModel` | 
| GraphSAGE | `SAGEModel` | 
| TransformerConv | `TransformerModel` |
| Graph Attention Network | `GATModel` | 
| Graph Isomorphism Network | `GINModel` |
| Relational GCN | `RGCNModel` |

## Output Files

| Script | Output | Description |
|---|---|---|
| `1_create_graph.py` | `graph_*.pt` | PyG `Data` object with `edge_type` tensor |
| `1_create_graph.py` | `graph_*_companions.pt` | `pos_edge_index`, `neg_edge_index`, negative pools, temporal split sets |
| `1_create_graph.py` | `graph_*_names.pt` | Drug / disease name lists and key-mappings |
| `1_create_graph.py` | `graph_*_analysis.json` | Graph statistics (with `--analyze`) |
| `2_train_models.py` | `{Model}_best_model_*.pt` | Best model weights (one file per architecture) |
| `2_train_models.py` | `training_summary_*.json` | Per-model paths, thresholds, validation metrics |
| `2_train_models.py` | `training_report_*.txt` | Human-readable training report |
| `2_train_models.py` | `*_training_curves_*.png` | Loss / AUC training curves per model |
| `3_test_evaluate.py` | `evaluation/test_results_summary_*.csv` | AUC, APR, F1, accuracy with bootstrap CIs |
| `3_test_evaluate.py` | `evaluation/test_results_detailed_*.json` | Full per-model metrics |
| `3_test_evaluate.py` | `predictions/{Model}_FP_predictions_*.csv` | False-positive candidates for explainer |
| `3_test_evaluate.py` | `figures/test_roc_curves_*.png` | ROC curves |
| `3_test_evaluate.py` | `figures/test_pr_curves_*.png` | Precision–Recall curves |
| `3_test_evaluate.py` | `figures/interactive_roc_curves_*.html` | Interactive Plotly ROC curves |
| `4_explainer.py` | `GNNExplainer_importance_{Model}_*.txt` | Bootstrap CI attribution report |
| `4_explainer.py` | `GNNExplainer_visualization_{Model}_*.html` | Interactive D3.js visualizer (Path + Model modes) |
| `4_explainer.py` | `GNNExplainer_node_importance_{Model}_*.csv` | Per-node degree-adjusted importance scores |
| `4_explainer.py` | `GNNExplainer_edge_importance_{Model}_*.csv` | Per-edge importance scores |

---

## License

MIT License
