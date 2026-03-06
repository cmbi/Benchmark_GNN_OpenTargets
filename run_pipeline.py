#!/usr/bin/env python3
"""
Complete Pipeline Runner for KG-Bench Drug-Disease Prediction
Runs the full modular pipeline: Graph -> Train -> Test -> Explain

Script interface summary
------------------------
1_create_graph.py   graph_path  [--config] [--output-dir] [--analyze]
2_train_models.py   graph_path  [--config] [--results-path] [--models]
3_test_evaluate.py  graph_path models_path [--config] [--results-path]
                                           [--export-fp] [--fp-threshold]
                                           [--fp-top-k]
4_explainer.py      graph_path results_path
                                [--mappings] [--summary] [--fp-csv]
                                [--model] [--top-diseases]
                                [--top-drugs-per-disease]
                                [--max-explanations] [--max-path-length]
                                [--max-paths-per-pair] [--n-bootstrap]
                                [--explainer-epochs] [--sample-size]
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
import datetime as dt


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f" ERROR in {description}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print(f" SUCCESS: {description}")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True


def _find_latest(directory, *pattern_parts):
    """Return the most-recently-modified file whose name contains all patterns."""
    try:
        files = [
            f for f in os.listdir(directory)
            if all(p.lower() in f.lower() for p in pattern_parts)
        ]
    except FileNotFoundError:
        return None
    if not files:
        return None
    files.sort(
        key=lambda f: os.path.getmtime(os.path.join(directory, f)),
        reverse=True,
    )
    return os.path.join(directory, files[0])


def run_complete_pipeline(config_path=None, results_dir="results"):
    """Run the complete drug-disease prediction pipeline."""

    print(" STARTING COMPLETE KG-BENCH DRUG-DISEASE PREDICTION PIPELINE")
    print("="*70)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{results_dir}/pipeline_run_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Create Graph
    # ------------------------------------------------------------------
    graph_cmd = [sys.executable, "scripts/1_create_graph.py",
                 "--output-dir", results_dir]
    if config_path:
        graph_cmd.extend(["--config", config_path])

    if not run_command(graph_cmd, "Graph Creation"):
        return False

    graph_files = list(Path(results_dir).glob("graph_*.pt"))
    # Exclude companion / names files
    graph_files = [p for p in graph_files
                   if "_companions" not in p.name and "_names" not in p.name]
    if not graph_files:
        print(" ERROR: No graph file found after creation")
        return False

    graph_path = str(graph_files[0])
    print(f" Using graph: {graph_path}")

    # ------------------------------------------------------------------
    # Step 2: Train Models
    # ------------------------------------------------------------------
    models_dir = f"{results_dir}/models"
    train_cmd = [sys.executable, "scripts/2_train_models.py", graph_path,
                 "--results-path", models_dir]
    if config_path:
        train_cmd.extend(["--config", config_path])

    if not run_command(train_cmd, "Model Training"):
        return False

    # ------------------------------------------------------------------
    # Step 3: Test and Evaluate Models
    # ------------------------------------------------------------------
    test_cmd = [sys.executable, "scripts/3_test_evaluate.py",
                graph_path, models_dir,
                "--results-path", results_dir, "--export-fp"]
    if config_path:
        test_cmd.extend(["--config", config_path])

    if not run_command(test_cmd, "Model Testing and Evaluation"):
        return False

    # ------------------------------------------------------------------
    # Step 4: Explain Predictions
    # 4_explainer.py positional args: graph_path  results_path
    # Named args: --mappings --summary --fp-csv --model ...
    # ------------------------------------------------------------------
    predictions_dir = Path(results_dir) / "predictions"
    summary_path = _find_latest(models_dir, "training_summary", ".json")
    mappings_path = (
        _find_latest("processed_data/mappings/", "all_mappings")
        or _find_latest(results_dir, "all_mappings")
    )

    if predictions_dir.exists():
        fp_files = list(predictions_dir.glob("*_FP_predictions_*.csv"))

        for fp_file in fp_files:
            # Derive the model name from the filename prefix
            model_name = fp_file.name.split("_FP_predictions_")[0]
            explainer_out_dir = f"{results_dir}/explainer/{model_name}"

            explain_cmd = [
                sys.executable, "scripts/4_explainer.py",
                graph_path,
                explainer_out_dir,
                "--model", model_name,
                "--fp-csv", str(fp_file),
            ]
            if summary_path and os.path.exists(summary_path):
                explain_cmd.extend(["--summary", summary_path])
            if mappings_path and os.path.exists(mappings_path):
                explain_cmd.extend(["--mappings", mappings_path])
            if config_path:
                explain_cmd.extend(["--config", config_path])

            run_command(explain_cmd, f"GNN Explanation for {model_name}")

    print(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" All results saved to: {results_dir}")
    print(f" Pipeline run timestamp: {timestamp}")
    return True


def run_individual_step(step, graph_path=None, models_path=None,
                        config_path=None, results_dir="results",
                        model_name=None, fp_csv=None,
                        mappings_path=None, summary_path=None):
    """Run an individual pipeline step."""

    print(f" RUNNING INDIVIDUAL STEP: {step.upper()}")
    print("="*50)

    if step == "graph":
        cmd = [sys.executable, "scripts/1_create_graph.py",
               "--output-dir", results_dir]
        if config_path:
            cmd.extend(["--config", config_path])
        return run_command(cmd, "Graph Creation")

    elif step == "train":
        if not graph_path:
            print(" ERROR: Graph path required for training")
            return False
        models_dir = f"{results_dir}/models"
        cmd = [sys.executable, "scripts/2_train_models.py", graph_path,
               "--results-path", models_dir]
        if config_path:
            cmd.extend(["--config", config_path])
        return run_command(cmd, "Model Training")

    elif step == "test":
        if not graph_path or not models_path:
            print(" ERROR: Graph path and models path required for testing")
            return False
        cmd = [sys.executable, "scripts/3_test_evaluate.py",
               graph_path, models_path,
               "--results-path", results_dir, "--export-fp"]
        if config_path:
            cmd.extend(["--config", config_path])
        return run_command(cmd, "Model Testing")

    elif step == "explain":
        if not graph_path:
            print(" ERROR: Graph path required for explain step")
            return False
        explainer_out_dir = (
            f"{results_dir}/explainer/{model_name}"
            if model_name else f"{results_dir}/explainer"
        )
        cmd = [sys.executable, "scripts/4_explainer.py",
               graph_path, explainer_out_dir]
        if model_name:
            cmd.extend(["--model", model_name])
        if fp_csv:
            cmd.extend(["--fp-csv", fp_csv])
        if summary_path:
            cmd.extend(["--summary", summary_path])
        if mappings_path:
            cmd.extend(["--mappings", mappings_path])
        if config_path:
            cmd.extend(["--config", config_path])
        return run_command(cmd, f"GNN Explanation{' for ' + model_name if model_name else ''}")

    else:
        print(f" ERROR: Unknown step '{step}'")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="KG-Bench Drug-Disease Prediction Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --complete

  # Run complete pipeline with custom config
  python run_pipeline.py --complete --config config.json

  # Run individual steps
  python run_pipeline.py --step graph
  python run_pipeline.py --step train  --graph results/pipeline_run_*/graph_*.pt
  python run_pipeline.py --step test   --graph results/pipeline_run_*/graph_*.pt \\
                                       --models results/pipeline_run_*/models/
  python run_pipeline.py --step explain --graph results/pipeline_run_*/graph_*.pt \\
                                        --model TransformerModel \\
                                        --fp-csv results/pipeline_run_*/predictions/*_FP_*.csv \\
                                        --summary results/pipeline_run_*/models/training_summary_*.json
        """
    )

    # Main execution modes
    parser.add_argument("--complete", action="store_true",
                        help="Run complete pipeline (graph -> train -> test -> explain)")
    parser.add_argument("--step", choices=["graph", "train", "test", "explain"],
                        help="Run individual pipeline step")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base results directory")

    # Step-specific arguments
    parser.add_argument("--graph", type=str,
                        help="Path to graph .pt file (for train / test / explain steps)")
    parser.add_argument("--models", type=str,
                        help="Path to models directory (for test step)")

    # Explain-step arguments
    parser.add_argument("--model", type=str, default="TransformerModel",
                        choices=["GCNModel", "SAGEModel", "TransformerModel",
                                 "GATModel", "GINModel", "RGCNModel"],
                        help="Which trained model to explain (explain step)")
    parser.add_argument("--fp-csv", type=str,
                        help="FP predictions CSV from step 3 (explain step; "
                             "auto-detected if omitted)")
    parser.add_argument("--summary", type=str,
                        help="Path to training_summary_*.json from step 2 "
                             "(explain step; auto-detected if omitted)")
    parser.add_argument("--mappings", type=str,
                        help="Path to all_mappings_*.pkl (explain step; "
                             "auto-detected if omitted)")

    # Utility flags
    parser.add_argument("--check-env", action="store_true",
                        help="Check if environment is properly set up")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment check
    # ------------------------------------------------------------------
    if args.check_env:
        print(" CHECKING ENVIRONMENT")
        print("="*30)
        print(f"Python version: {sys.version}")

        scripts = [
            "1_create_graph.py",
            "2_train_models.py",
            "3_test_evaluate.py",
            "4_explainer.py",
        ]
        for script in scripts:
            path = f"scripts/{script}"
            print(f"{'' if os.path.exists(path) else ''} {path}")

        for module in ["models.py", "utils.py", "config.py", "data_processing.py"]:
            path = f"src/{module}"
            print(f"{'' if os.path.exists(path) else ''} {path}")

        for pkg, label in [("torch", "PyTorch"),
                            ("torch_geometric", "PyTorch Geometric"),
                            ("sklearn", "scikit-learn"),
                            ("networkx", "NetworkX")]:
            try:
                mod = __import__(pkg)
                print(f" {label}: {mod.__version__}")
            except ImportError:
                print(f" {label} not installed")
        return

    # ------------------------------------------------------------------
    # Validate mode
    # ------------------------------------------------------------------
    if not args.complete and not args.step:
        print(" ERROR: Must specify either --complete or --step")
        parser.print_help()
        return

    if args.complete and args.step:
        print(" ERROR: Cannot specify both --complete and --step")
        return

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    try:
        if args.complete:
            success = run_complete_pipeline(args.config, args.results_dir)
        else:
            success = run_individual_step(
                step=args.step,
                graph_path=args.graph,
                models_path=args.models,
                config_path=args.config,
                results_dir=args.results_dir,
                model_name=args.model,
                fp_csv=args.fp_csv,
                mappings_path=args.mappings,
                summary_path=args.summary,
            )

        if success:
            print("\n EXECUTION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n EXECUTION FAILED!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
