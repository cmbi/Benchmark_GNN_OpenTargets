# Drug-Disease Prediction Pipeline

A machine learning pipeline for predicting drug-disease associations using Graph Neural Networks (GNNs). Implements GCN, GraphSAGE, and Graph Transformer models.

Identifiers:



## Project Structure

```
drug_disease_prediction/
├── 1_graph_creation.py          # Data loading and graph construction
├── 2_training_validation.py     # Model training and validation
├── 3_testing_evaluation.py      # Model testing and evaluation
├── run_pipeline.py              # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── config_example.json          # Example configuration file
└── data/                        # Data directory (create this)
    ├── raw/                     # Raw OpenTargets data
    └── processed/               # Processed data files
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd drug_disease_prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Choose one of the following options:

### Option 1: Download Raw OpenTargets Data (Complete Setup)

Visit the OpenTargets downloads page to access the data: https://platform.opentargets.org/downloads/

#### Using FileZilla (Recommended)
1. **Host**: `ftp.ebi.ac.uk`
2. **Remote site**: `/pub/databases/opentargets/platform/`
3. **Navigate** to the version folders: `21.06`, `23.06`, or `24.06`
4. **Go to**: `output/etl/parquet/` within each version
5. **Download** the required datasets from each version

#### Command Line Download
```bash
# Create directory structure
mkdir -p data/raw/{21.06,23.06,24.06}

# Download using wget (example for 21.06)
cd data/raw/21.06
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
data/raw/
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

### Option 2: Generate Pre-processed Data (Quick Start)

For a faster setup, you can directly use the processed files

The data structure:
```
data/processed/  (or your configured processed_path)
├── tables/
│   ├── processed_molecules.csv     # Filtered drug molecules
│   ├── processed_indications.csv   # Drug-disease indications
│   ├── processed_diseases.csv      # Filtered diseases
│   ├── processed_genes.csv         # Target genes
│   └── processed_associations.csv  # Gene-disease associations
├── mappings/
│   ├── drug_key_mapping.json       # Drug ID to node index
│   ├── drug_type_key_mapping.json  # Drug type mappings
│   ├── gene_key_mapping.json       # Gene ID mappings
│   ├── reactome_key_mapping.json   # Pathway mappings
│   ├── disease_key_mapping.json    # Disease ID mappings
│   ├── therapeutic_area_key_mapping.json # Therapeutic area mappings
│   └── mapping_summary.json        # Node count summary
└── edges/
    ├── 1_molecule_drugType_edges.pt   # Drug-DrugType edges
    ├── 2_molecule_disease_edges.pt    # Drug-Disease edges  
    ├── 3_molecule_gene_edges.pt       # Drug-Gene edges
    ├── 4_gene_reactome_edges.pt       # Gene-Pathway edges
    ├── 5_disease_therapeutic_edges.pt # Disease-TherapeuticArea edges
    ├── 6_disease_gene_edges.pt        # Disease-Gene edges
    ├── edge_statistics.json           # Edge count summary
    └── training_drug_disease_pairs.csv # Training pairs with names
```

#### Quick Start with Generated Data
Once the processing script completes:

```bash
# The processed data is ready for graph creation
python 1_graph_creation.py

# Or run the complete pipeline
python run_pipeline.py
```

## Usage

### Complete Pipeline
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Step 1: Create graph (skip if using pre-processed data)
python 1_graph_creation.py

# Step 2: Train models
python 2_training_validation.py <graph_path> <results_path>

# Step 3: Evaluate models
python 3_testing_evaluation.py <graph_path> <models_info_path> <results_path>
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
  "general_path": "data/raw/",
  "results_path": "results/"
}
```

## Models

- **GCN**: Graph Convolutional Network
- **GraphSAGE**: Sample and Aggregate
- **Graph Transformer**: Attention-based GNN

## Output Files

- `*_graph.pt` - Graph objects
- `*_best_model.pt` - Trained models
- `test_results_summary.csv` - Performance metrics
- `test_evaluation_report.txt` - Detailed results

## License

MIT License
