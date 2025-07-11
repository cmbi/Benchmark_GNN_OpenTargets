"""
Main Pipeline Script for Drug-Disease Prediction
This script orchestrates the complete pipeline: graph creation, training, and evaluation.
"""

import os
import sys
import argparse
import json
import datetime as dt
from pathlib import Path

# Import our modules
from graph_creation import create_graph, get_config
from training_validation import train_all_models
from testing_evaluation import run_evaluation

def setup_directories(base_path="drug_disease_prediction"):
    """Setup directory structure for the project."""
    
    directories = {
        'base': base_path,
        'results': f"{base_path}/results",
        'models': f"{base_path}/models", 
        'graphs': f"{base_path}/graphs",
        'reports': f"{base_path}/reports",
        'data': f"{base_path}/data"
    }
    
    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return directories

def save_config(config, file_path):
    """Save configuration to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_config(file_path):
    """Load configuration from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def run_complete_pipeline(config_path=None, skip_graph=False, skip_training=False, skip_evaluation=False):
    """Run the complete pipeline from start to finish."""
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Drug-Disease Prediction Pipeline - {timestamp}")
    print("="*80)
    
    # Setup directories
    directories = setup_directories()
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = load_config(config_path)
    else:
        print("Using default configuration")
        config = get_config()
        config['results_path'] = directories['results'] + "/"
        
        # Save default config for future reference
        config_file = f"{directories['base']}/config_{timestamp}.json"
        save_config(config, config_file)
        print(f"Saved configuration to {config_file}")
    
    # Initialize results storage
    pipeline_results = {
        'timestamp': timestamp,
        'config': config,
        'graph_path': None,
        'trained_models': None,
        'test_results': None
    }
    
    # Step 1: Graph Creation
    if not skip_graph:
        print("\n" + "="*60)
        print("STEP 1: GRAPH CREATION")
        print("="*60)
        
        try:
            graph, graph_path, builder = create_graph(config)
            pipeline_results['graph_path'] = graph_path
            
            # Save builder for later use
            builder_path = f"{directories['graphs']}/builder_{timestamp}.pt"
            import torch
            torch.save(builder, builder_path)
            pipeline_results['builder_path'] = builder_path
            
            print(f"✓ Graph creation completed successfully!")
            print(f"  Graph saved to: {graph_path}")
            print(f"  Builder saved to: {builder_path}")
            
        except Exception as e:
            print(f"✗ Graph creation failed: {str(e)}")
            return None
    else:
        print("Skipping graph creation...")
        # You would need to provide existing graph path
        pipeline_results['graph_path'] = input("Enter path to existing graph file: ")
    
    # Step 2: Model Training and Validation
    if not skip_training:
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING & VALIDATION")
        print("="*60)
        
        try:
            # Load builder if available
            builder = None
            if 'builder_path' in pipeline_results:
                import torch
                builder = torch.load(pipeline_results['builder_path'])
            
            trained_models, validation_results = train_all_models(
                pipeline_results['graph_path'], 
                directories['results'] + "/",
                builder
            )
            
            pipeline_results['trained_models'] = trained_models
            pipeline_results['validation_results'] = validation_results
            
            # Save trained models info
            models_info_path = f"{directories['models']}/trained_models_{timestamp}.json"
            
            # Convert model info to serializable format
            models_info = {}
            for model_name, model_data in trained_models.items():
                models_info[model_name] = {
                    'model_path': model_data['model_path'],
                    'threshold': model_data['threshold'],
                    'validation_auc': model_data['validation_auc']
                }
            
            save_config(models_info, models_info_path)
            pipeline_results['models_info_path'] = models_info_path
            
            print(f"✓ Training completed successfully!")
            print(f"  Models info saved to: {models_info_path}")
            
            # Print validation summary
            print("\nValidation Results Summary:")
            for model_name, results in validation_results.items():
                print(f"  {model_name}: AUC = {results['metrics']['auc']:.4f}")
            
        except Exception as e:
            print(f"✗ Training failed: {str(e)}")
            return None
    else:
        print("Skipping training...")
        pipeline_results['models_info_path'] = input("Enter path to trained models info file: ")
    
    # Step 3: Testing and Evaluation
    if not skip_evaluation:
        print("\n" + "="*60)
        print("STEP 3: TESTING & EVALUATION")
        print("="*60)
        
        try:
            test_results = run_evaluation(
                pipeline_results['graph_path'],
                pipeline_results.get('models_info_path'),
                directories['reports'] + "/"
            )
            
            pipeline_results['test_results'] = test_results
            
            print(f"✓ Evaluation completed successfully!")
            print(f"  Reports saved to: {directories['reports']}/")
            
            # Print test summary
            print("\nTest Results Summary:")
            for model_name, results in test_results.items():
                metrics = results['metrics']
                print(f"  {model_name}:")
                print(f"    AUC: {metrics['auc']:.4f}")
                print(f"    F1:  {metrics['f1']:.4f}")
                print(f"    Acc: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"✗ Evaluation failed: {str(e)}")
            return None
    else:
        print("Skipping evaluation...")
    
    # Save complete pipeline results
    results_file = f"{directories['base']}/pipeline_results_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = {
        'timestamp': pipeline_results['timestamp'],
        'config': pipeline_results['config'],
        'graph_path': pipeline_results['graph_path'],
        'models_info_path': pipeline_results.get('models_info_path'),
        'builder_path': pipeline_results.get('builder_path')
    }
    
    # Add test results summary if available
    if pipeline_results.get('test_results'):
        serializable_results['test_summary'] = {}
        for model_name, results in pipeline_results['test_results'].items():
            serializable_results['test_summary'][model_name] = results['metrics']
    
    save_config(serializable_results, results_file)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Complete results saved to: {results_file}")
    print(f"All outputs available in: {directories['base']}/")
    
    return pipeline_results

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Drug-Disease Prediction Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--skip-graph', action='store_true', help='Skip graph creation step')
    parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation step')
    parser.add_argument('--graph-only', action='store_true', help='Run only graph creation')
    parser.add_argument('--train-only', action='store_true', help='Run only training (requires existing graph)')
    parser.add_argument('--eval-only', action='store_true', help='Run only evaluation (requires existing models)')
    
    args = parser.parse_args()
    
    # Handle specific run modes
    if args.graph_only:
        args.skip_training = True
        args.skip_evaluation = True
    elif args.train_only:
        args.skip_graph = True
        args.skip_evaluation = True
    elif args.eval_only:
        args.skip_graph = True
        args.skip_training = True
    
    try:
        results = run_complete_pipeline(
            config_path=args.config,
            skip_graph=args.skip_graph,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation
        )
        
        if results is None:
            print("Pipeline failed!")
            sys.exit(1)
        else:
            print("Pipeline completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()