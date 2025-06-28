#!/usr/bin/env python3
"""
Multi-Dataset Evaluation Wrapper
Runs model evaluation across all supported datasets automatically.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import argparse


class MultiDatasetEvaluator:
    def __init__(self):
        # All supported datasets based on the project knowledge
        self.datasets = [
            "cifar10",
            "cifar100",
            "food101",
            "fgcv_aircraft",
            "oxford_pets",
            "caltech256"
        ]

        # Valid evaluation scripts
        self.valid_scripts = [
            "evaluate_model.py",
            "evaluate_models.py",
            "evaluate_models_kd.py"
        ]

    def validate_inputs(self, script_name, model_path):
        """Validate input parameters"""
        # Check if evaluation script exists
        if not os.path.exists(script_name):
            raise FileNotFoundError(f"Evaluation script '{script_name}' not found")

        # Check if it's a valid evaluation script
        script_basename = os.path.basename(script_name)
        if script_basename not in self.valid_scripts:
            print(f"Warning: '{script_basename}' is not a recognized evaluation script")
            print(f"Expected one of: {self.valid_scripts}")

        # Check if model checkpoint exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint '{model_path}' not found")

    def run_single_evaluation(self, script_name, model_path, dataset, additional_args=None):
        """Run evaluation on a single dataset"""
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset.upper()} dataset")
        print(f"{'=' * 60}")

        # Build command
        cmd = [
            sys.executable, script_name,
            "--model_checkpoint", model_path,
            "--dataset", dataset
        ]

        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)

        print(f"Running command: {' '.join(cmd)}")

        start_time = time.time()

        # Set environment variables to handle Unicode issues on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '1'

        try:
            # Run the evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"✓ {dataset} evaluation completed successfully in {duration:.2f}s")

            return {
                "dataset": dataset,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time

            print(f"✗ {dataset} evaluation failed after {duration:.2f}s")
            print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")

            return {
                "dataset": dataset,
                "status": "failed",
                "duration": duration,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }

    def parse_results_from_output(self, output_text):
        """Extract accuracy results from the output text"""
        results = {}
        lines = output_text.split('\n')

        for line in lines:
            line = line.strip()

            # Try different patterns for accuracy results
            patterns = [
                # Pattern: "Top-1 Accuracy: 0.1505"
                (r"Top-1 Accuracy:\s*([0-9]*\.?[0-9]+)", "top_1_accuracy"),
                (r"Top-5 Accuracy:\s*([0-9]*\.?[0-9]+)", "top_5_accuracy"),

                # Pattern: "Results on CIFAR-10:" followed by accuracy lines
                (r"Top-1 Accuracy:\s*([0-9]*\.?[0-9]+)", "top_1_accuracy"),
                (r"Top-5 Accuracy:\s*([0-9]*\.?[0-9]+)", "top_5_accuracy"),

                # Pattern: Just numbers or percentages
                (r"Top-1:\s*([0-9]*\.?[0-9]+)", "top_1_accuracy"),
                (r"Top-5:\s*([0-9]*\.?[0-9]+)", "top_5_accuracy"),

                # Pattern: Dictionary format like {'top_1_accuracy': 0.1505, 'top_5_accuracy': 0.5339}
                (r"'top_1_accuracy':\s*([0-9]*\.?[0-9]+)", "top_1_accuracy"),
                (r"'top_5_accuracy':\s*([0-9]*\.?[0-9]+)", "top_5_accuracy"),

                # Pattern: Accuracy results in JSON-like format
                (r'"top_1_accuracy":\s*([0-9]*\.?[0-9]+)', "top_1_accuracy"),
                (r'"top_5_accuracy":\s*([0-9]*\.?[0-9]+)', "top_5_accuracy"),
            ]

            import re
            for pattern, key in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        # Convert to percentage if it's a decimal (0.0-1.0 range)
                        if value <= 1.0:
                            value *= 100
                        results[key] = value
                    except ValueError:
                        continue

        # Also try to parse the entire output as JSON if it looks like JSON
        try:
            import json
            # Look for JSON-like structures in the output
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        for key in ['top_1_accuracy', 'top_5_accuracy']:
                            if key in parsed:
                                value = float(parsed[key])
                                if value <= 1.0:
                                    value *= 100
                                results[key] = value
        except (json.JSONDecodeError, ValueError):
            pass

        return results

    def run_all_evaluations(self, script_name, model_path, additional_args=None,
                            datasets=None, save_results=True):
        """Run evaluations on all datasets"""
        # Use provided datasets or default to all
        datasets_to_run = datasets if datasets else self.datasets

        print(f"Starting multi-dataset evaluation...")
        print(f"Script: {script_name}")
        print(f"Model: {model_path}")
        print(f"Datasets: {', '.join(datasets_to_run)}")
        print(f"Additional args: {additional_args}")

        overall_start_time = time.time()
        all_results = []
        summary_results = {}

        for dataset in datasets_to_run:
            result = self.run_single_evaluation(
                script_name, model_path, dataset, additional_args
            )
            all_results.append(result)

            # Parse accuracy results if successful
            if result["status"] == "success":
                accuracy_results = self.parse_results_from_output(result["stdout"])
                summary_results[dataset] = accuracy_results

                # Debug: Print what we found in the output
                if not accuracy_results:
                    print(f"   ⚠️  No accuracy results found in output for {dataset}")
                    # Print last few lines of stdout for debugging
                    stdout_lines = result["stdout"].strip().split('\n')
                    print(f"   Last 5 lines of output:")
                    for line in stdout_lines[-5:]:
                        if line.strip():
                            print(f"   > {line}")
                else:
                    print(f"   📊 Found results: {accuracy_results}")

        overall_end_time = time.time()
        total_duration = overall_end_time - overall_start_time

        # Print summary
        self.print_summary(summary_results, total_duration)

        # Save results if requested
        if save_results:
            self.save_results(script_name, model_path, all_results, summary_results, total_duration)

        return all_results, summary_results

    def print_summary(self, summary_results, total_duration):
        """Print evaluation summary"""
        print(f"\n{'=' * 80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total evaluation time: {total_duration:.2f}s ({total_duration / 60:.1f} minutes)")
        print(
            f"Successful evaluations: {len([dataset for dataset in self.datasets if dataset in summary_results and summary_results[dataset]])}/{len(self.datasets)}")
        print(f"\nResults by dataset:")
        print(f"{'Dataset':<15} {'Top-1 Acc (%)':<15} {'Top-5 Acc (%)':<15} {'Status':<10}")
        print(f"{'-' * 60}")

        for dataset in self.datasets:
            if dataset in summary_results and summary_results[dataset]:
                results = summary_results[dataset]
                top1 = f"{results.get('top_1_accuracy', 0):.2f}" if 'top_1_accuracy' in results else "N/A"
                top5 = f"{results.get('top_5_accuracy', 0):.2f}" if 'top_5_accuracy' in results else "N/A"
                status = "✓"
            else:
                top1 = "N/A"
                top5 = "N/A"
                status = "✗"

            print(f"{dataset:<15} {top1:<15} {top5:<15} {status:<10}")

    def save_results(self, script_name, model_path, all_results, summary_results, total_duration):
        """Save results to JSON file in results folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        script_basename = os.path.splitext(os.path.basename(script_name))[0]

        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Create filename with model name and timestamp
        results_filename = os.path.join(results_dir, f"{model_name}_evaluation_{timestamp}.json")

        results_data = {
            "metadata": {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "script": script_name,
                "script_basename": script_basename,
                "model": model_path,
                "model_name": model_name,
                "total_duration": total_duration,
                "total_duration_minutes": round(total_duration / 60, 2),
                "datasets_evaluated": list(summary_results.keys()),
                "successful_evaluations": len([r for r in all_results if r["status"] == "success"]),
                "failed_evaluations": len([r for r in all_results if r["status"] == "failed"])
            },
            "summary_results": summary_results,
            "detailed_results": all_results
        }

        # Save main results file
        with open(results_filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Also save a compact summary CSV for easy viewing
        csv_filename = os.path.join(results_dir, f"{model_name}_summary_{timestamp}.csv")
        self.save_summary_csv(csv_filename, summary_results, model_name, timestamp)

        print(f"\n📊 Results saved to:")
        print(f"   📄 Detailed: {results_filename}")
        print(f"   📈 Summary:  {csv_filename}")

    def save_summary_csv(self, csv_filename, summary_results, model_name, timestamp):
        """Save a compact CSV summary of results"""
        import csv

        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['Model', 'Timestamp', 'Dataset', 'Top-1 Accuracy (%)', 'Top-5 Accuracy (%)', 'Status'])

            # Write results
            for dataset in self.datasets:
                if dataset in summary_results and summary_results[dataset]:
                    results = summary_results[dataset]
                    top1 = f"{results.get('top_1_accuracy', 0):.2f}" if 'top_1_accuracy' in results else "N/A"
                    top5 = f"{results.get('top_5_accuracy', 0):.2f}" if 'top_5_accuracy' in results else "N/A"
                    status = "Success"
                else:
                    top1 = "N/A"
                    top5 = "N/A"
                    status = "Failed/No Results"

                writer.writerow([model_name, timestamp, dataset, top1, top5, status])


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python eval_wrapper.py --script evaluate_models.py --model checkpoints/mobileclip_kd.pth

  # With config file
  python eval_wrapper.py --script evaluate_models.py --model model.ckpt --config_path config.yaml

  # With specific datasets only
  python eval_wrapper.py --script evaluate_model.py --model model.pth --datasets cifar10 cifar100

  # With additional arguments
  python eval_wrapper.py --script evaluate_models_kd.py --model model.pth --batch_size 32 --prompt "A photo of a"
        """
    )

    parser.add_argument(
        "--script",
        required=True,
        help="Path to evaluation script (evaluate_model.py, evaluate_models.py, or evaluate_models_kd.py)"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Specific datasets to evaluate (default: all supported datasets)"
    )

    parser.add_argument(
        "--config_path",
        help="Path to config file"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--prompt",
        help="Prompt template for evaluation"
    )

    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to file"
    )

    # Parse known args to allow passing through additional arguments
    args, unknown_args = parser.parse_known_args()

    # Build additional arguments list
    additional_args = []
    if args.config_path:
        additional_args.extend(["--config_path", args.config_path])
    if args.batch_size:
        additional_args.extend(["--batch_size", str(args.batch_size)])
    if args.prompt:
        additional_args.extend(["--prompt", args.prompt])

    # Add any unknown arguments (pass-through)
    additional_args.extend(unknown_args)

    try:
        evaluator = MultiDatasetEvaluator()

        # Validate inputs
        evaluator.validate_inputs(args.script, args.model)

        # Run evaluations
        all_results, summary_results = evaluator.run_all_evaluations(
            script_name=args.script,
            model_path=args.model,
            additional_args=additional_args if additional_args else None,
            datasets=args.datasets,
            save_results=not args.no_save
        )

        # Check if any evaluations failed
        failed_count = sum(1 for r in all_results if r["status"] == "failed")
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} evaluation(s) failed. Check the detailed output above.")
            sys.exit(1)
        else:
            print(f"\n🎉 All evaluations completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()