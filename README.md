# Drug-Disease Prediction Pipeline

A machine learning pipeline for predicting drug-disease associations using Graph Neural Networks (GNNs). Implements GCN, GraphSAGE, and Graph Transformer models.

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

### Download OpenTargets Data

Visit the OpenTargets downloads page to access the data: https://platform.opentargets.org/downloads/

#### Using FileZilla (Recommended)
1. **Host**: `ftp.ebi.ac.uk`
2. **Remote site**: `/pub/databases/opentargets/platform/`
3. **Navigate** to the version folders: `21.06`, `23.06`, or `24.06`
4. **Download** the required datasets from each version

#### Command Line Download
```bash
# Create directory structure
mkdir -p data/raw/{21.06,23.06,24.06}

# Download using wget (example for 21.06)
cd data/raw/21.06
wget -r -np -nH --cut-dirs=5 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/drug_indication/
wget -r -np -nH --cut-dirs=5 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/molecule/
wget -r -np -nH --cut-dirs=5 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/diseases/
wget -r -np -nH --cut-dirs=5 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/targets/
wget -r -np -nH --cut-dirs=5 https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/21.06/output/associationByOverallDirect/

# Repeat for versions 23.06 and 24.06 (only drug_indication needed for these)
```

### Required Data by Version

#### Training Version (21.06):
From `/pub/databases/opentargets/platform/21.06/output/`:
- `drug_indication/` → rename to `indication/`
- `molecule/`
- `diseases/`
- `targets/`
- `associationByOverallDirect/`

#### Validation Version (23.06):
From `/pub/databases/opentargets/platform/23.06/output/`:
- `drug_indication/` → rename to `indication/`

#### Test Version (24.06):
From `/pub/databases/opentargets/platform/24.06/output/`:
- `drug_indication/` → rename to `indication/`

### Final Directory Structure:
```
data/raw/
├── 21.06/
│   ├── indication/           # renamed from drug_indication
│   ├── molecule/            
│   ├── diseases/            
│   ├── targets/             
│   └── associationByOverallDirect/
├── 23.06/
│   └── indication/          # renamed from drug_indication
└── 24.06/
    └── indication/          # renamed from drug_indication
```

**Important Notes:**
- All files are in PARQUET format
- Rename `drug_indication` folders to `indication` after download
- Large datasets may require significant download time and storage space
- Check OpenTargets license terms before using the data

## Usage

### Complete Pipeline
```bash
python run_pipeline.py
```

### Individual Steps
```bash
# Step 1: Create graph
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
