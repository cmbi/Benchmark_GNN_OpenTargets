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

Download OpenTargets data and place in the following structure:
- `data/raw/{version}/indication/`
- `data/raw/{version}/molecule/`
- `data/raw/{version}/diseases/`
- `data/raw/{version}/targets/`
- `data/raw/{version}/{association_type}/`

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
