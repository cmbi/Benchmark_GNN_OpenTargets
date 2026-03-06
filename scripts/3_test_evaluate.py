#!/usr/bin/env python3
"""
Model Testing and Evaluation Script for Drug-Disease Prediction
Comprehensive testing with FP export for GNNExplainer analysis.
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
import argparse
from pathlib import Path
import random

# Import from shared modules
from src.models import GCNModel, TransformerModel, SAGEModel, MODEL_CLASSES
from src.utils import set_seed, enable_full_reproducibility, calculate_metrics, calculate_bootstrap_ci
from src.config import get_config, create_custom_config


class ModelEvaluator:
    """Comprehensive model evaluation with FP export for GNNExplainer."""

    def __init__(self, results_path="results/"):
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dirs = self._create_results_structure()

    def _create_results_structure(self):
        """Create organized results directory structure."""
        subdirs = {
            'evaluation': 'evaluation',
            'predictions': 'predictions',
            'explainer': 'explainer',
            'figures': 'figures',
            'logs': 'logs'
        }

        result_dirs = {}
        for key, dirname in subdirs.items():
            dir_path = os.path.join(self.results_path, dirname)
            os.makedirs(dir_path, exist_ok=True)
            result_dirs[key] = dir_path

        print(f"Results directory structure created under: {self.results_path}")
        return result_dirs

    def _load_training_summary(self, models_dir):
        """
        Find and load the most recent training_summary_*.json in models_dir.
        Returns the parsed dict, or an empty dict if none is found.
        """
        try:
            summary_files = [
                f for f in os.listdir(models_dir)
                if 'training_summary' in f and f.endswith('.json')
            ]
        except FileNotFoundError:
            return {}

        if not summary_files:
            return {}

        summary_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
            reverse=True
        )
        summary_path = os.path.join(models_dir, summary_files[0])
        print(f"Loading training summary: {summary_path}")
        with open(summary_path) as f:
            return json.load(f)

    def load_trained_models_info(self, models_path):
        """
        Load information about trained models.

        Architecture parameters (num_layers, hidden_channels) are read from
        training_summary_*.json when available, so the model reconstructed
        here matches exactly what was saved during training.
        """
        trained_models = {}

        # Locate model weight files
        if os.path.isfile(models_path):
            model_files = [models_path]
            models_dir = os.path.dirname(models_path)
        else:
            models_dir = models_path
            model_files = []
            for f in os.listdir(models_dir):
                if f.endswith('_best_model.pt') or f.endswith('.pt'):
                    model_files.append(os.path.join(models_dir, f))

        print(f"Found model files: {[os.path.basename(f) for f in model_files]}")

        # Build initial entries with fallback defaults
        for model_file in model_files:
            filename = os.path.basename(model_file)

            if 'GCN' in filename:
                model_name = 'GCNModel'
                model_class = GCNModel
            elif 'Transformer' in filename:
                model_name = 'TransformerModel'
                model_class = TransformerModel
            elif 'SAGE' in filename:
                model_name = 'SAGEModel'
                model_class = SAGEModel
            else:
                model_name = filename.replace('_best_model.pt', '').replace('.pt', '')
                model_class = MODEL_CLASSES.get(model_name, TransformerModel)

            trained_models[model_name] = {
                'model_path':      model_file,
                'model_class':     model_class,
                'threshold':       0.5,
                'num_layers':      2,   # fallback — overridden below
                'hidden_channels': 16,  # fallback — overridden below
            }

        # Override architecture params from training_summary if available
        summary = self._load_training_summary(models_dir)
        model_results = summary.get('model_results', {})

        for model_name in trained_models:
            best = model_results.get(model_name, {})

            # training_summary stores these under 'num_layers' and 'hidden_dim'
            if 'num_layers' in best:
                trained_models[model_name]['num_layers'] = best['num_layers']
                print(f"  {model_name}: num_layers = {best['num_layers']} (from training_summary)")
            else:
                print(f"  {model_name}: num_layers not found in training_summary, "
                      f"using fallback = {trained_models[model_name]['num_layers']}")

            if 'hidden_dim' in best:
                trained_models[model_name]['hidden_channels'] = best['hidden_dim']
                print(f"  {model_name}: hidden_channels = {best['hidden_dim']} (from training_summary)")
            else:
                print(f"  {model_name}: hidden_dim not found in training_summary, "
                      f"using fallback = {trained_models[model_name]['hidden_channels']}")

        return trained_models

    def create_test_data_from_graph(self, graph, use_test_split=True):
        """Create test dataset from the graph object."""
        print("Creating test dataset from graph...")

        if use_test_split and hasattr(graph, 'test_edge_index') and hasattr(graph, 'test_edge_label'):
            print("Using existing test split from graph")
            test_edge_tensor = graph.test_edge_index
            test_label_tensor = graph.test_edge_label
        elif hasattr(graph, 'val_edge_index') and hasattr(graph, 'val_edge_label'):
            print("Using validation split as test set")
            test_edge_tensor = graph.val_edge_index
            test_label_tensor = graph.val_edge_label
        else:
            print("Creating synthetic test dataset...")
            num_test_edges = min(1000, graph.edge_index.size(1) // 10)

            perm = torch.randperm(graph.edge_index.size(1))
            test_pos_indices = perm[:num_test_edges // 2]
            test_pos_edges = graph.edge_index[:, test_pos_indices].t()

            num_nodes = graph.x.size(0)
            test_neg_edges = []
            while len(test_neg_edges) < num_test_edges // 2:
                src = random.randint(0, num_nodes - 1)
                dst = random.randint(0, num_nodes - 1)
                if src != dst:
                    test_neg_edges.append([src, dst])

            test_neg_edges = torch.tensor(test_neg_edges, dtype=torch.long)

            test_edge_tensor = torch.cat([test_pos_edges, test_neg_edges], dim=0)
            test_label_tensor = torch.cat([
                torch.ones(len(test_pos_edges), dtype=torch.long),
                torch.zeros(len(test_neg_edges), dtype=torch.long)
            ])

        print(f"Test set contains {len(test_edge_tensor)} samples")
        print(f"Positive samples: {torch.sum(test_label_tensor == 1).item()}")
        print(f"Negative samples: {torch.sum(test_label_tensor == 0).item()}")

        return test_edge_tensor, test_label_tensor

    def test_single_model(self, model_info, graph, test_edge_tensor, test_label_tensor):
        """Test a single model and return detailed results."""
        model_path    = model_info['model_path']
        model_class   = model_info['model_class']
        threshold     = model_info['threshold']
        # Read architecture params saved by 2_train_models.py
        num_layers      = model_info.get('num_layers', 2)
        hidden_channels = model_info.get('hidden_channels', 16)

        print(f"Testing model from {model_path}")
        print(f"  Architecture: num_layers={num_layers}, hidden_channels={hidden_channels}")

        # Reconstruct model with the same architecture used during training
        model = model_class(
            in_channels=graph.x.size(1),
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout_rate=0.5
        ).to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Move data to device
        graph             = graph.to(self.device)
        test_edge_tensor  = test_edge_tensor.to(self.device)
        test_label_tensor = test_label_tensor.to(self.device)

        # Make predictions in batches to avoid memory issues
        batch_size = 1000
        test_probs = []

        with torch.no_grad():
            z = model(graph.x.float(), graph.edge_index)

            for start in range(0, len(test_edge_tensor), batch_size):
                end         = min(start + batch_size, len(test_edge_tensor))
                batch_edges = test_edge_tensor[start:end]
                batch_scores = (z[batch_edges[:, 0]] * z[batch_edges[:, 1]]).sum(dim=-1)
                batch_probs  = torch.sigmoid(batch_scores)
                test_probs.append(batch_probs.cpu().numpy())

        test_probs    = np.concatenate(test_probs)
        test_preds    = (test_probs >= threshold).astype(int)
        test_labels_np = test_label_tensor.cpu().numpy()

        metrics = calculate_metrics(test_labels_np, test_probs, test_preds)

        return {
            'model':         model,
            'probabilities': test_probs,
            'predictions':   test_preds,
            'labels':        test_labels_np,
            'metrics':       metrics,
            'test_edges':    test_edge_tensor.cpu().numpy(),
            'threshold':     threshold
        }

    def export_fp_predictions(self, model_name, model_results, graph,
                               threshold=0.7, top_k=10000):
        """Export False Positive predictions for GNNExplainer analysis."""
        print(f"\nExporting FP predictions for {model_name}...")

        probabilities = model_results['probabilities']
        labels        = model_results['labels']
        test_edges    = model_results['test_edges']

        fp_mask    = (labels == 0) & (probabilities >= threshold)
        fp_indices = np.where(fp_mask)[0]

        if len(fp_indices) == 0:
            print(f"No FP predictions found above threshold {threshold}")
            return None

        print(f"Found {len(fp_indices)} FP predictions above threshold {threshold}")

        fp_scores      = probabilities[fp_indices]
        sorted_indices = np.argsort(fp_scores)[::-1]
        top_fp_indices = fp_indices[sorted_indices[:top_k]]
        top_fp_scores  = fp_scores[sorted_indices[:top_k]]

        print(f"Exporting top {len(top_fp_indices)} FP predictions")

        fp_data = []
        for i, (fp_idx, score) in enumerate(zip(top_fp_indices, top_fp_scores)):
            edge_idx     = int(fp_idx)
            drug_idx     = int(test_edges[edge_idx, 0])
            disease_idx  = int(test_edges[edge_idx, 1])
            drug_name    = f"Drug_{drug_idx}"
            disease_name = f"Disease_{disease_idx}"

            fp_data.append({
                'rank':           i + 1,
                'drug_idx':       drug_idx,
                'disease_idx':    disease_idx,
                'drug_name':      drug_name,
                'disease_name':   disease_name,
                'confident_score': float(score),
                'prediction':     float(score),
                'true_label':     0,
                'model':          model_name
            })

        fp_df     = pd.DataFrame(fp_data)
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

        # 1. CSV (primary format for GNNExplainer)
        csv_filename = f"{model_name}_FP_predictions_{timestamp}.csv"
        csv_path     = os.path.join(self.results_dirs['predictions'], csv_filename)
        fp_df.to_csv(csv_path, index=False)
        print(f"FP predictions saved as CSV: {csv_path}")

        # 2. PyTorch tensor (backup)
        tensor_data = [[row['drug_name'], row['disease_name'], row['confident_score']]
                       for _, row in fp_df.iterrows()]
        pt_filename = f"{model_name}_FP_predictions_{timestamp}.pt"
        pt_path     = os.path.join(self.results_dirs['predictions'], pt_filename)
        torch.save(tensor_data, pt_path)
        print(f"FP predictions saved as PyTorch: {pt_path}")

        # 3. Summary statistics
        summary_data = {
            'model_name':          model_name,
            'timestamp':           timestamp,
            'total_test_edges':    len(test_edges),
            'total_negative_edges': int(np.sum(labels == 0)),
            'fp_threshold':        threshold,
            'top_k':               top_k,
            'total_fp_candidates': len(fp_indices),
            'exported_fp_count':   len(fp_data),
            'confidence_range':    [float(fp_df['confident_score'].min()),
                                    float(fp_df['confident_score'].max())],
            'mean_confidence':     float(fp_df['confident_score'].mean()),
            'csv_file':            csv_filename,
            'pt_file':             pt_filename
        }

        summary_filename = f"{model_name}_FP_export_summary_{timestamp}.json"
        summary_path     = os.path.join(self.results_dirs['predictions'], summary_filename)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Export summary saved: {summary_path}")

        return {
            'csv_path':    csv_path,
            'pt_path':     pt_path,
            'summary_path': summary_path,
            'fp_count':    len(fp_data),
            'summary':     summary_data
        }

    def test_all_models(self, trained_models_info, graph, test_edge_tensor, test_label_tensor,
                        export_fp=True, fp_threshold=0.7, fp_top_k=10000):
        """Test all trained models and return comprehensive results."""

        test_results = {}

        for model_name, model_info in trained_models_info.items():
            print(f"\nTesting {model_name}...")

            try:
                results = self.test_single_model(
                    model_info, graph, test_edge_tensor, test_label_tensor)
                test_results[model_name] = results

                if export_fp:
                    fp_results = self.export_fp_predictions(
                        model_name=model_name,
                        model_results=results,
                        graph=graph,
                        threshold=fp_threshold,
                        top_k=fp_top_k
                    )
                    if fp_results:
                        results['fp_export'] = fp_results
                        print(f"FP export successful: {fp_results['fp_count']} predictions exported")

                metrics    = results['metrics']
                ci_results = metrics.get('ci_results', {})

                print(f"\nTest Results for {model_name}:")
                print("-" * 40)
                print(f"AUC: {metrics['auc']:.4f}")
                if 'auc' in ci_results:
                    ci = ci_results['auc']
                    print(f"     (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")

                print(f"APR: {metrics['apr']:.4f}")
                if 'apr' in ci_results:
                    ci = ci_results['apr']
                    print(f"     (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")

                print(f"F1-Score: {metrics['f1']:.4f}")
                if 'f1' in ci_results:
                    ci = ci_results['f1']
                    print(f"         (95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}])")

                print(f"Accuracy: {metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                continue

        return test_results

    def create_visualizations(self, test_results):
        """Create comprehensive visualizations of test results."""

        datetime_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")

        self._plot_roc_curves(test_results, datetime_str)
        self._plot_pr_curves(test_results, datetime_str)
        self._plot_confusion_matrices(test_results, datetime_str)
        self._plot_metrics_comparison(test_results, datetime_str)
        self._create_interactive_plots(test_results, datetime_str)

    def _plot_roc_curves(self, test_results, datetime_str):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(12, 10))

        colors      = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']

        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc     = auc(fpr, tpr)
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
        plt.savefig(f'{self.results_dirs["figures"]}/test_roc_curves_{datetime_str}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pr_curves(self, test_results, datetime_str):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(12, 10))

        colors      = ['blue', 'red', 'green', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':']

        for i, (model_name, results) in enumerate(test_results.items()):
            precision, recall, _ = precision_recall_curve(
                results['labels'], results['probabilities'])
            avg_precision = average_precision_score(
                results['labels'], results['probabilities'])
            plt.plot(recall, precision,
                     color=colors[i % len(colors)],
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=3,
                     label=f'{model_name} (AP = {avg_precision:.4f})')

        baseline = np.mean(test_results[list(test_results.keys())[0]]['labels'])
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Random (AP = {baseline:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dirs["figures"]}/test_pr_curves_{datetime_str}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, test_results, datetime_str):
        """Plot confusion matrices for all models."""
        n_models = len(test_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

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
        plt.savefig(f'{self.results_dirs["figures"]}/test_confusion_matrices_{datetime_str}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_comparison(self, test_results, datetime_str):
        """Plot comprehensive metrics comparison."""
        metrics_to_plot = ['auc', 'apr', 'f1', 'accuracy', 'precision', 'recall', 'specificity']
        model_names     = list(test_results.keys())

        metrics_data = {metric: [] for metric in metrics_to_plot}
        for model_name in model_names:
            for metric in metrics_to_plot:
                metrics_data[metric].append(test_results[model_name]['metrics'][metric])

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes      = axes.flatten()
        colors    = ['steelblue', 'coral', 'lightgreen', 'gold',
                     'mediumpurple', 'lightcoral', 'lightskyblue']

        for i, metric in enumerate(metrics_to_plot):
            values = metrics_data[metric]
            bars   = axes[i].bar(model_names, values, color=colors[i % len(colors)], alpha=0.8)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            axes[i].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_ylim(0, 1.1)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)

        fig.delaxes(axes[7])
        plt.suptitle('Model Performance Comparison - Test Set', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_dirs["figures"]}/test_metrics_comparison_{datetime_str}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_interactive_plots(self, test_results, datetime_str):
        """Create interactive Plotly visualizations."""
        fig_roc = go.Figure()
        colors  = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (model_name, results) in enumerate(test_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc     = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.4f})',
                line=dict(color=colors[i % len(colors)], width=3)
            ))

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
        fig_roc.write_html(
            f'{self.results_dirs["figures"]}/interactive_roc_curves_{datetime_str}.html')

    def save_detailed_results(self, test_results):
        """Save detailed test results to files."""
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

        summary_data = []
        for model_name, results in test_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            row['Threshold'] = results['threshold']

            ci_results = results['metrics'].get('ci_results', {})
            for metric in ['auc', 'apr', 'f1', 'precision', 'sensitivity', 'specificity']:
                if metric in ci_results:
                    ci = ci_results[metric]
                    row[f'{metric}_ci_lower'] = ci['ci_lower']
                    row[f'{metric}_ci_upper'] = ci['ci_upper']
                    row[f'{metric}_ci_width']  = ci['ci_width']

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            f'{self.results_dirs["evaluation"]}/test_results_summary_{timestamp}.csv',
            index=False)

        results_for_json = {}
        for model_name, results in test_results.items():
            results_for_json[model_name] = {
                'metrics':              results['metrics'],
                'threshold':            results['threshold'],
                'predictions_sample':   results['predictions'][:100].tolist(),
                'probabilities_sample': results['probabilities'][:100].tolist()
            }
            if 'fp_export' in results:
                results_for_json[model_name]['fp_export_summary'] = results['fp_export']['summary']

        with open(f'{self.results_dirs["evaluation"]}/test_results_detailed_{timestamp}.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)

        print(f"Detailed results saved to: {self.results_dirs['evaluation']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test and evaluate trained models')
    parser.add_argument('graph_path',   help='Path to graph file (.pt)')
    parser.add_argument('models_path',  help='Path to trained models directory or single model file')
    parser.add_argument('--results-path',  default='results/', help='Results output directory')
    parser.add_argument('--export-fp',     action='store_true', default=True,
                        help='Export FP predictions for GNNExplainer')
    parser.add_argument('--fp-threshold',  type=float, default=0.7, help='FP confidence threshold')
    parser.add_argument('--fp-top-k',      type=int,   default=10000,
                        help='Maximum FP predictions to export')
    parser.add_argument('--config',        type=str,   help='Path to config JSON file')

    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = create_custom_config(**config_dict)
    else:
        config = get_config()

    enable_full_reproducibility(42)

    evaluator = ModelEvaluator(args.results_path)

    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")

    graph = torch.load(args.graph_path, map_location=evaluator.device)
    print(f"Loaded graph: {graph}")

    trained_models_info = evaluator.load_trained_models_info(args.models_path)

    if not trained_models_info:
        print("No trained models found! Please train models first.")
        return None

    print(f"Found trained models: {list(trained_models_info.keys())}")

    test_edge_tensor, test_label_tensor = evaluator.create_test_data_from_graph(graph)

    test_results = evaluator.test_all_models(
        trained_models_info,
        graph,
        test_edge_tensor,
        test_label_tensor,
        export_fp=args.export_fp,
        fp_threshold=args.fp_threshold,
        fp_top_k=args.fp_top_k
    )

    if not test_results:
        print("No successful test results!")
        return None

    print("Creating visualizations...")
    evaluator.create_visualizations(test_results)

    print("Saving detailed results...")
    evaluator.save_detailed_results(test_results)

    print(f"\nEvaluation completed! Results saved to {args.results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, results in test_results.items():
        metrics = results['metrics']
        print(f"\n{model_name}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1:  {metrics['f1']:.4f}")
        print(f"  Acc: {metrics['accuracy']:.4f}")

        if 'fp_export' in results:
            fp_count = results['fp_export']['fp_count']
            csv_file = results['fp_export']['summary']['csv_file']
            print(f"  FP Exported: {fp_count} predictions")
            print(f"  CSV File: {args.results_path}predictions/{csv_file}")

    if args.export_fp:
        print("\n" + "=" * 60)
        print("GNNEXPLAINER INTEGRATION")
        print("=" * 60)
        print("To analyze FP predictions with GNNExplainer, run:")

        for model_name, results in test_results.items():
            if 'fp_export' in results:
                csv_file = results['fp_export']['summary']['csv_file']
                print(f"\n# For {model_name}:")
                print(f"python scripts/4_explainer.py \\")
                print(f"  {args.graph_path} \\")
                print(f"  {args.results_path}explainer/{model_name} \\")
                print(f"  --model {model_name} \\")
                print(f"  --fp-csv {args.results_path}predictions/{csv_file}")

    return test_results


if __name__ == "__main__":
    main()
