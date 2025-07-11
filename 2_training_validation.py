"""
Training and Validation Module for Drug-Disease Prediction
This module handles model definitions, training, and validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import random
import numpy as np
import datetime as dt

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model Definitions
class GCNModel(torch.nn.Module):
    """Graph Convolutional Network model."""
    
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

class TransformerModel(torch.nn.Module):
    """Graph Transformer model."""
    
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

class SAGEModel(torch.nn.Module):
    """GraphSAGE model."""
    
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

class Trainer:
    """Training and validation handler."""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _get_default_config(self):
        """Get default training configuration."""
        return {
            'learning_rate': 0.0005,
            'hidden_channels': 16,
            'out_channels': 16,
            'num_layers': 2,
            'dropout_rate': 0.5,
            'num_epochs': 1000,
            'patience': 10,
            'batch_size': 1000
        }
    
    def prepare_training_data(self, graph, builder):
        """Prepare positive and negative training edges."""
        # Extract positive edges (existing drug-disease connections)
        molecule_disease_edges = builder.molecule_disease_edges
        existing_drug_disease_edges = list(zip(molecule_disease_edges[0].tolist(), molecule_disease_edges[1].tolist()))
        pos_edge_index = torch.tensor(existing_drug_disease_edges).T
        
        # Generate negative samples
        all_molecule_disease = builder.generate_pairs(
            builder.approved_drugs_list, 
            builder.disease_list, 
            builder.drug_key_mapping, 
            builder.disease_key_mapping
        )
        
        existing_edges_set = set(existing_drug_disease_edges)
        not_linked_set = list(set(all_molecule_disease) - existing_edges_set)
        
        # Sample negative edges
        random.seed(42)
        num_neg_samples = len(existing_drug_disease_edges)
        neg_edges = random.sample(not_linked_set, num_neg_samples)
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).T
        
        return pos_edge_index, neg_edge_index
    
    def train_single_model(self, model, graph, pos_edge_index, neg_edge_index, 
                          val_edge_tensor, val_label_tensor, results_path, model_name):
        """Train a single model with early stopping."""
        model = model.to(self.device)
        graph = graph.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)
        val_edge_tensor = val_edge_tensor.to(self.device)
        val_label_tensor = val_label_tensor.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        loss_function = torch.nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        best_val_auc = 0.0
        counter = 0
        best_threshold = 0.5
        
        # For saving the best model
        datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        best_model_path = f'{results_path}{model_name}_best_model_{datetime_str}.pt'
        
        print(f"Training {model_name}...")
        
        for epoch in tqdm(range(self.config['num_epochs']), desc=f'Training {model_name}'):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            z = model(graph.x.float(), graph.edge_index)
            
            # Compute scores for positive and negative edges
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
            
            # Validation phase
            if epoch % 5 == 0:  # Validate every 5 epochs
                model.eval()
                with torch.no_grad():
                    z = model(graph.x.float(), graph.edge_index)
                    val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
                    val_loss = loss_function(val_scores, val_label_tensor.float())
                    val_probs = torch.sigmoid(val_scores)
                    
                    # Calculate validation AUC
                    val_auc = roc_auc_score(val_label_tensor.cpu().numpy(), val_probs.cpu().numpy())
                    val_threshold = val_probs.mean().item()
                    
                    # Check for improvement
                    if val_auc > best_val_auc:
                        best_val_loss = val_loss
                        best_val_auc = val_auc
                        best_threshold = val_threshold
                        counter = 0
                        
                        # Save best model
                        torch.save(model.state_dict(), best_model_path)
                        
                        if epoch % 50 == 0:
                            print(f"Epoch {epoch+1}: New best validation AUC: {best_val_auc:.4f}, Loss: {best_val_loss:.4f}")
                    else:
                        counter += 1
                    
                    # Early stopping
                    if counter >= self.config['patience']:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
        
        print(f"Training completed for {model_name}")
        print(f"Best validation AUC: {best_val_auc:.4f}")
        print(f"Best threshold: {best_threshold:.4f}")
        
        return best_model_path, best_threshold, best_val_auc
    
    def validate_model(self, model_path, model_class, graph, val_edge_tensor, val_label_tensor, threshold):
        """Validate a trained model."""
        # Load the best model
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=self.config['hidden_channels'],
            out_channels=self.config['out_channels'],
            num_layers=self.config['num_layers'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        graph = graph.to(self.device)
        val_edge_tensor = val_edge_tensor.to(self.device)
        val_label_tensor = val_label_tensor.to(self.device)
        
        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)
            val_scores = (z[val_edge_tensor[:, 0]] * z[val_edge_tensor[:, 1]]).sum(dim=-1)
            val_probs = torch.sigmoid(val_scores)
            val_preds = (val_probs >= threshold).float()
            
            # Calculate metrics
            val_auc = roc_auc_score(val_label_tensor.cpu().numpy(), val_probs.cpu().numpy())
            val_apr = average_precision_score(val_label_tensor.cpu().numpy(), val_probs.cpu().numpy())
            
            # Calculate accuracy, precision, recall, F1
            TP = ((val_preds == 1) & (val_label_tensor == 1)).sum().item()
            FP = ((val_preds == 1) & (val_label_tensor == 0)).sum().item()
            TN = ((val_preds == 0) & (val_label_tensor == 0)).sum().item()
            FN = ((val_preds == 0) & (val_label_tensor == 1)).sum().item()
            
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            validation_metrics = {
                'auc': val_auc,
                'apr': val_apr,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
            }
        
        return validation_metrics, val_probs.cpu().numpy(), val_preds.cpu().numpy()

def train_all_models(graph_path, results_path, builder=None):
    """Train all model types and return trained models with validation results."""
    
    # Load graph
    graph = torch.load(graph_path)
    print(f"Loaded graph: {graph}")
    
    # Initialize trainer
    trainer = Trainer()
    
    # Prepare training data
    if builder is None:
        # If builder is not provided, we need to recreate some essential data
        # This is a simplified version - in practice you'd save and load this data
        print("Warning: Builder not provided. Some functionality may be limited.")
        pos_edge_index = graph.edge_index  # This is a simplification
        neg_edge_index = graph.edge_index  # This needs proper negative sampling
    else:
        pos_edge_index, neg_edge_index = trainer.prepare_training_data(graph, builder)
    
    # Extract validation data from graph
    val_edge_tensor = graph.val_edge_index
    val_label_tensor = graph.val_edge_label
    
    # Define models to train
    models_to_train = {
        'GCNModel': GCNModel,
        'TransformerModel': TransformerModel,
        'SAGEModel': SAGEModel
    }
    
    # Train all models
    trained_models = {}
    validation_results = {}
    
    for model_name, model_class in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize model
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=trainer.config['hidden_channels'],
            out_channels=trainer.config['out_channels'],
            num_layers=trainer.config['num_layers'],
            dropout_rate=trainer.config['dropout_rate']
        )
        
        # Train model
        best_model_path, best_threshold, best_val_auc = trainer.train_single_model(
            model, graph, pos_edge_index, neg_edge_index,
            val_edge_tensor, val_label_tensor, results_path, model_name
        )
        
        # Validate model
        val_metrics, val_probs, val_preds = trainer.validate_model(
            best_model_path, model_class, graph,
            val_edge_tensor, val_label_tensor, best_threshold
        )
        
        # Store results
        trained_models[model_name] = {
            'model_path': best_model_path,
            'model_class': model_class,
            'threshold': best_threshold,
            'validation_auc': best_val_auc
        }
        
        validation_results[model_name] = {
            'metrics': val_metrics,
            'probabilities': val_probs,
            'predictions': val_preds,
            'threshold': best_threshold
        }
        
        print(f"Validation Results for {model_name}:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  APR: {val_metrics['apr']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Save training summary
    datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    summary_path = f"{results_path}training_summary_{datetime_str}.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Training and Validation Summary\n")
        f.write("===============================\n\n")
        
        for model_name, results in validation_results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model Path: {trained_models[model_name]['model_path']}\n")
            f.write(f"Best Threshold: {results['threshold']:.4f}\n")
            f.write(f"Validation Metrics:\n")
            f.write(f"  AUC: {results['metrics']['auc']:.4f}\n")
            f.write(f"  APR: {results['metrics']['apr']:.4f}\n")
            f.write(f"  Accuracy: {results['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['metrics']['precision']:.4f}\n")
            f.write(f"  Recall: {results['metrics']['recall']:.4f}\n")
            f.write(f"  F1-Score: {results['metrics']['f1']:.4f}\n")
            
            cm = results['metrics']['confusion_matrix']
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TP: {cm['TP']}, FP: {cm['FP']}\n")
            f.write(f"    FN: {cm['FN']}, TN: {cm['TN']}\n\n")
        
        # Model ranking
        f.write("Model Ranking by Validation AUC:\n")
        sorted_models = sorted(validation_results.items(), 
                             key=lambda x: x[1]['metrics']['auc'], reverse=True)
        for i, (model_name, results) in enumerate(sorted_models):
            f.write(f"{i+1}. {model_name}: {results['metrics']['auc']:.4f}\n")
    
    print(f"\nTraining summary saved to: {summary_path}")
    
    return trained_models, validation_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
        results_path = sys.argv[2] if len(sys.argv) > 2 else "results/"
    else:
        print("Usage: python 2_training_validation.py <graph_path> [results_path]")
        sys.exit(1)
    
    trained_models, validation_results = train_all_models(graph_path, results_path)