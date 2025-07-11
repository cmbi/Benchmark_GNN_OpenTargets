"""
Testing and Evaluation Module for Drug-Disease Prediction
This module handles model testing, performance evaluation, and visualization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score, classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
import json
import os
from training_validation import GCNModel, TransformerModel, SAGEModel

class ModelEvaluator:
    """Class for comprehensive model evaluation and testing."""
    
    def __init__(self, results_path="results/"):
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_test_data(self, graph, builder, training_version=21.06, test_version=24.06):
        """Create test dataset from future data."""
        print("Creating test dataset...")
        
        # This would typically load test data from a different time period
        # For now, we'll simulate this by creating a held-out test set
        
        # Extract existing training edges
        existing_edges = set()
        if hasattr(builder, 'molecule_disease_edges'):
            molecule_disease_edges = builder.molecule_disease_edges
            existing_edges = set(zip(molecule_disease_edges[0].tolist(), molecule_disease_edges[1].tolist()))
        
        # Get validation edges
        val_edges = set()
        if hasattr(graph, 'val_edge_index'):
            val_tensor = graph.val_edge_index
            val_edges = set(zip(val_tensor[:, 0].tolist(), val_tensor[:, 1].tolist()))
        
        # Generate all possible drug-disease pairs
        if hasattr(builder, 'approved_drugs_list') and hasattr(builder, 'disease_list'):
            all_possible_pairs = []
            for i, drug in enumerate(builder.approved_drugs_list):
                for j, disease in enumerate(builder.disease_list):
                    drug_idx = builder.drug_key_mapping[drug]
                    disease_idx = builder.disease_key_mapping[disease]
                    all_possible_pairs.append((drug_idx, disease_idx))
        else:
            # Fallback: generate test pairs from graph structure
            num_drugs = len([i for i in range(graph.x.size(0)) if graph.x[i, 0] == 1])  # Assuming drug one-hot is first
            num_diseases = len([i for i in range(graph.x.size(0)) if graph.x[i, 3] == 1])  # Assuming disease one-hot is 4th
            
            all_possible_pairs = []
            for i in range(num_drugs):
                for j in range(num_drugs, num_drugs + num_diseases):
                    all_possible_pairs.append((i, j))
        
        # Remove training and validation edges to get potential test edges
        available_pairs = set(all_possible_pairs) - existing_edges - val_edges
        available_pairs = list(available_pairs)
        
        # Sample test edges (simulate future approvals)
        np.random.seed(42)
        test_size = min(1000, len(available_pairs) // 2)
        
        # Create balanced test set
        positive_test_pairs = np.random.choice(len(available_pairs), test_size // 2, replace=False)
        test_positive_edges = [available_pairs[i] for i in positive_test_pairs]
        
        # Remove selected positive edges from available pairs
        remaining_pairs = [pair for i, pair in enumerate(available_pairs) if i not in positive_test_pairs]
        negative_test_pairs = np.random.choice(len(remaining_pairs), test_size // 2, replace=False)
        test_negative_edges = [remaining_pairs[i] for i in negative_test_pairs]
        
        # Combine and create labels
        test_edges = test_positive_edges + test_negative_edges
        test_labels = [1] * len(test_positive_edges) + [0] * len(test_negative_edges)
        
        # Convert to tensors
        test_edge_tensor = torch.tensor(test_edges, dtype=torch.long)
        test_label_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        print(f"Created test set with {len(test_positive_edges)} positive and {len(test_negative_edges)} negative samples")
        
        return test_edge_tensor, test_label_tensor
    
    def test_model(self, model_path, model_class, graph, test_edge_tensor, test_label_tensor, threshold):
        """Test a single model and return detailed results."""
        
        # Load model
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=16,
            out_channels=16,
            num_layers=2,
            dropout_rate=0.5
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Move data to device
        graph = graph.to(self.device)
        test_edge_tensor = test_edge_tensor.to(self.device)
        test_label_tensor = test_label_tensor.to(self.device)
        
        # Make predictions
        batch_size = 1000
        test_probs = []
        
        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)
            
            # Process in batches to avoid memory issues
            for start in range(0, len(test_edge_tensor), batch_size):
                end = min(start + batch_size, len(test_edge_tensor))
                batch_edges = test_edge_tensor[start:end]
                
                # Calculate edge scores
                batch_scores = (z[batch_edges[:, 0]] * z[batch_edges[:, 1]]).sum(dim=-1)
                batch_probs = torch.sigmoid(batch_scores)
                test_probs.append(batch_probs.cpu().numpy())
        
        # Combine results
        test_probs = np.concatenate(test_probs)
        test_preds = (test_probs >= threshold).astype(int)
        test_labels_np = test_label_tensor.cpu().numpy()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(test_labels_np, test_probs, test_preds)
        
        return {
            'probabilities': test_probs,
            'predictions': test_preds,
            'labels': test_labels_np,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, y_true, y_prob, y_pred):
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        auc_score = roc_auc_score(y_true, y_prob)
        apr_score = average_precision_score(y_true, y_prob)
        
        # Additional metrics
        ppv = precision  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc': auc_score,
            'apr': apr_score,
            'ppv': ppv,
            'npv': npv,
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp), 
                'TN': int(tn), 'FN': int(fn)
            }
        }
    
    def test_all_models(self, trained_models, graph, test_edge_tensor, test_label_tensor):
        """Test all trained models and return comprehensive results."""
        
        test_results = {}
        
        for model_name, model_info in trained_models.items():
            print(f"Testing {model_name}...")
            
            # Get model class
            model_classes = {
                'GCNModel': GCNModel,
                'TransformerModel': TransformerModel,
                'SAGEModel': SAGEModel
            }
            
            model_class = model_classes[model_name]
            
            # Test model
            results = self.test_model(
                model_info['model_path'],
                model_class,
                graph,
                test_edge_tensor,
                test_label_tensor,
                model_info['threshold']
            )
            
            results['threshold'] = model_info['threshold']
            test_results[model_name] = results
            
            # Print results
            metrics = results['metrics']
            print(f"Test Results for {model_name}:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  APR: {metrics['apr']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print()
        
        return test_results
    
    def create_visualizations(self, test_results):
        """Create comprehensive visualizations of test results."""
        
        datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 1. ROC Curves
        self._plot_roc_curves(test_results, datetime_str)
        
        # 2. Precision-Recall Curves
        self._plot_pr_curves(test_results, datetime_str)
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices(test_results, datetime_str)
        
        # 4. Metrics Comparison
        self._plot_metrics_comparison(test_results, datetime_str)
        
        # 5. Interactive Plotly Visualizations
        self._create_interactive_plots(test_results, datetime_str)
        
    def _plot_roc_curves(self, test_results, datetime_str):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(12, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=3,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}test_confusion_matrices_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, test_results, datetime_str):
        """Plot comprehensive metrics comparison."""
        metrics_to_plot = ['auc', 'apr', 'f1', 'accuracy', 'precision', 'recall', 'specificity']
        model_names = list(test_results.keys())
        
        # Prepare data
        metrics_data = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            for metric in metrics_to_plot:
                metrics_data[metric].append(test_results[model_name]['metrics'][metric])
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'mediumpurple', 'lightcoral', 'lightskyblue']
        
        for i, metric in enumerate(metrics_to_plot):
            values = metrics_data[metric]
            bars = axes[i].bar(model_names, values, color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove the last empty subplot
        fig.delaxes(axes[7])
        
        plt.suptitle('Model Performance Comparison - Test Set', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}test_metrics_comparison_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_plots(self, test_results, datetime_str):
        """Create interactive Plotly visualizations."""
        
        # Interactive ROC Curves
        fig_roc = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.4f})',
                line=dict(color=colors[i % len(colors)], width=3)
            ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='black', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            title='Interactive ROC Curves - Test Set',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600
        )
        
        fig_roc.write_html(f'{self.results_path}interactive_roc_curves_{datetime_str}.html')
        
        # Interactive Metrics Radar Chart
        metrics_names = ['AUC', 'APR', 'F1', 'Accuracy', 'Precision', 'Recall', 'Specificity']
        
        fig_radar = go.Figure()
        
        for i, (model_name, results) in enumerate(test_results.items()):
            metrics_values = [
                results['metrics']['auc'],
                results['metrics']['apr'],
                results['metrics']['f1'],
                results['metrics']['accuracy'],
                results['metrics']['precision'],
                results['metrics']['recall'],
                results['metrics']['specificity']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=metrics_values,
                theta=metrics_names,
                fill='toself',
                name=model_name,
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart - Test Set",
            width=800, height=600
        )
        
        fig_radar.write_html(f'{self.results_path}interactive_radar_chart_{datetime_str}.html')
    
    def save_detailed_results(self, test_results, datetime_str):
        """Save detailed test results to files."""
        
        # Save metrics summary
        summary_data = []
        for model_name, results in test_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            row['Threshold'] = results['threshold']
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.results_path}test_results_summary_{datetime_str}.csv', index=False)
        summary_df.to_excel(f'{self.results_path}test_results_summary_{datetime_str}.xlsx', index=False)
        
        # Save detailed results as JSON
        results_for_json = {}
        for model_name, results in test_results.items():
            results_for_json[model_name] = {
                'metrics': results['metrics'],
                'threshold': results['threshold'],
                'predictions_sample': results['predictions'][:100].tolist(),  # First 100 predictions
                'probabilities_sample': results['probabilities'][:100].tolist()  # First 100 probabilities
            }
        
        with open(f'{self.results_path}test_results_detailed_{datetime_str}.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Create comprehensive report
        self._create_report(test_results, datetime_str)
    
    def _create_report(self, test_results, datetime_str):
        """Create a comprehensive text report."""
        
        report_path = f'{self.results_path}test_evaluation_report_{datetime_str}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Find best performing model
            best_auc_model = max(test_results.items(), key=lambda x: x[1]['metrics']['auc'])
            best_f1_model = max(test_results.items(), key=lambda x: x[1]['metrics']['f1'])
            
            f.write(f"Best AUC Performance: {best_auc_model[0]} ({best_auc_model[1]['metrics']['auc']:.4f})\n")
            f.write(f"Best F1 Performance: {best_f1_model[0]} ({best_f1_model[1]['metrics']['f1']:.4f})\n")
            f.write(f"Total Models Evaluated: {len(test_results)}\n\n")
            
            # Detailed Results for Each Model
            for model_name, results in test_results.items():
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("=" * 50 + "\n")
                
                metrics = results['metrics']
                cm = metrics['confusion_matrix']
                
                f.write(f"Threshold Used: {results['threshold']:.4f}\n\n")
                
                f.write("Performance Metrics:\n")
                f.write(f"  • AUC-ROC: {metrics['auc']:.4f}\n")
                f.write(f"  • Average Precision: {metrics['apr']:.4f}\n")
                f.write(f"  • Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  • Precision (PPV): {metrics['precision']:.4f}\n")
                f.write(f"  • Recall (Sensitivity): {metrics['recall']:.4f}\n")
                f.write(f"  • Specificity: {metrics['specificity']:.4f}\n")
                f.write(f"  • F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"  • NPV: {metrics['npv']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"  True Positives:  {cm['TP']:4d}\n")
                f.write(f"  False Positives: {cm['FP']:4d}\n")
                f.write(f"  True Negatives:  {cm['TN']:4d}\n")
                f.write(f"  False Negatives: {cm['FN']:4d}\n\n")
                
                # Clinical Interpretation
                f.write("Clinical Interpretation:\n")
                if metrics['precision'] > 0.8:
                    f.write("  • High precision - Low false positive rate\n")
                elif metrics['precision'] > 0.6:
                    f.write("  • Moderate precision - Acceptable false positive rate\n")
                else:
                    f.write("  • Low precision - High false positive rate\n")
                
                if metrics['recall'] > 0.8:
                    f.write("  • High recall - Low false negative rate\n")
                elif metrics['recall'] > 0.6:
                    f.write("  • Moderate recall - Acceptable false negative rate\n")
                else:
                    f.write("  • Low recall - High false negative rate\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
            
            # Model Comparison
            f.write("MODEL COMPARISON\n")
            f.write("=" * 40 + "\n")
            
            # Ranking by different metrics
            metrics_for_ranking = ['auc', 'apr', 'f1', 'accuracy']
            
            for metric in metrics_for_ranking:
                f.write(f"\nRanking by {metric.upper()}:\n")
                sorted_models = sorted(test_results.items(), 
                                     key=lambda x: x[1]['metrics'][metric], reverse=True)
                for i, (model_name, results) in enumerate(sorted_models):
                    f.write(f"  {i+1}. {model_name}: {results['metrics'][metric]:.4f}\n")
            
            # Recommendations
            f.write("\nRECOMMENDations:\n")
            f.write("=" * 40 + "\n")
            
            if best_auc_model[1]['metrics']['auc'] > 0.8:
                f.write("• Excellent model performance achieved\n")
            elif best_auc_model[1]['metrics']['auc'] > 0.7:
                f.write("• Good model performance achieved\n")
            else:
                f.write("• Model performance needs improvement\n")
            
            f.write(f"• Consider {best_auc_model[0]} for deployment based on AUC performance\n")
            f.write(f"• Consider {best_f1_model[0]} for balanced precision-recall performance\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

def run_evaluation(graph_path, trained_models_path=None, results_path="results/"):
    """Main function to run complete evaluation."""
    
    print("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(results_path)
    
    # Load graph
    graph = torch.load(graph_path, map_location=evaluator.device)
    print(f"Loaded graph: {graph}")
    
    # Load trained models
    if trained_models_path:
        with open(trained_models_path, 'r') as f:
            trained_models = json.load(f)
    else:
        # Default trained models structure (you might need to adapt this)
        trained_models = {
            'GCNModel': {
                'model_path': f'{results_path}GCNModel_best_model.pt',
                'threshold': 0.5
            },
            'TransformerModel': {
                'model_path': f'{results_path}TransformerModel_best_model.pt',
                'threshold': 0.5
            },
            'SAGEModel': {
                'model_path': f'{results_path}SAGEModel_best_model.pt',
                'threshold': 0.5
            }
        }
    
    # Create test data
    test_edge_tensor, test_label_tensor = evaluator.create_test_data(graph, None)
    
    # Test all models
    test_results = evaluator.test_all_models(trained_models, graph, test_edge_tensor, test_label_tensor)
    
    # Create visualizations
    print("Creating visualizations...")
    evaluator.create_visualizations(test_results)
    
    # Save detailed results
    datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    print("Saving detailed results...")
    evaluator.save_detailed_results(test_results, datetime_str)
    
    print(f"Evaluation completed! Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, results in test_results.items():
        metrics = results['metrics']
        print(f"\n{model_name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1:  {metrics['f1']:.4f}")
        print(f"  Acc: {metrics['accuracy']:.4f}")
    
    return test_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python 3_testing_evaluation.py <graph_path> [trained_models_path] [results_path]")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    trained_models_path = sys.argv[2] if len(sys.argv) > 2 else None
    results_path = sys.argv[3] if len(sys.argv) > 3 else "results/"
    
    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)
    
    test_results = run_evaluation(graph_path, trained_models_path, results_path)savefig(f'{self.results_path}test_roc_curves_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curves(self, test_results, datetime_str):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(12, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']
        
        for i, (model_name, results) in enumerate(test_results.items()):
            precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])
            avg_precision = average_precision_score(results['labels'], results['probabilities'])
            
            plt.plot(recall, precision,
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=3,
                    label=f'{model_name} (AP = {avg_precision:.4f})')
        
        # Baseline
        baseline = np.mean(test_results[list(test_results.keys())[0]]['labels'])
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.7, label=f'Random (AP = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}test_pr_curves_{datetime_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, test_results, datetime_str):
        """Plot confusion matrices for all models."""
        n_models = len(test_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(test_results.items()):
            cm = confusion_matrix(results['labels'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}', fontsize=14)
            axes[i].set_xlabel('Predicted Label', fontsize=12)
            axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_xticklabels(['Negative', 'Positive'])
            axes[i].set_yticklabels(['Negative', 'Positive'])
        
        plt.tight_layout()
        plt.