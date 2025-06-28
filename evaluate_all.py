#!/usr/bin/env python3
"""
Fixed Batch Evaluation Script for MobileCLIP Knowledge Distillation
Compatible with existing evaluation scripts, handles Windows encoding issues
"""

import os
import subprocess
import json
import pandas as pd
import argparse
import re
from pathlib import Path
import time
from datetime import datetime
import sys

# Fix Windows encoding issues
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'


class FixedBatchEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Create timestamp-specific directory
        self.batch_results_dir = self.results_dir / f"batch_eval_{self.timestamp}"
        self.batch_results_dir.mkdir(exist_ok=True)

        print(f"Results will be saved to: {self.batch_results_dir}")

    def parse_accuracy_from_output(self, output_text):
        """Parse accuracy values from stdout text"""
        accuracy_data = {"top1": 0.0, "top5": 0.0}

        # Look for your specific format: {'top_1_accuracy': 0.1505, 'top_5_accuracy': 0.5339}
        import re

        # Pattern for dictionary format
        dict_pattern = r"['\"]top_1_accuracy['\"]:\s*([0-9.]+)"
        match = re.search(dict_pattern, output_text)
        if match:
            value = float(match.group(1))
            accuracy_data["top1"] = value * 100 if value <= 1.0 else value

        dict_pattern_top5 = r"['\"]top_5_accuracy['\"]:\s*([0-9.]+)"
        match = re.search(dict_pattern_top5, output_text)
        if match:
            value = float(match.group(1))
            accuracy_data["top5"] = value * 100 if value <= 1.0 else value

        # If dictionary format didn't work, try other patterns
        if accuracy_data["top1"] == 0.0:
            patterns = [
                r"Top-1[^:]*:\s*([0-9.]+)%?",
                r"Top1[^:]*:\s*([0-9.]+)%?",
                r"top1[^:]*:\s*([0-9.]+)%?",
                r"Accuracy[^:]*:\s*([0-9.]+)%?",
                r"accuracy[^:]*:\s*([0-9.]+)%?"
            ]

            for pattern in patterns:
                match = re.search(pattern, output_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Convert to percentage if needed
                    if value <= 1.0:
                        value *= 100
                    accuracy_data["top1"] = value
                    break

        # Look for Top-5 patterns if not found in dictionary
        if accuracy_data["top5"] == 0.0:
            top5_patterns = [
                r"Top-5[^:]*:\s*([0-9.]+)%?",
                r"Top5[^:]*:\s*([0-9.]+)%?",
                r"top5[^:]*:\s*([0-9.]+)%?"
            ]

            for pattern in top5_patterns:
                match = re.search(pattern, output_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    if value <= 1.0:
                        value *= 100
                    accuracy_data["top5"] = value
                    break

        return accuracy_data

    def run_base_model_evaluation(self, dataset):
        """Run evaluation for base MobileCLIP model"""
        # Clean dataset name for safe file operations
        clean_dataset = dataset.strip().lower().replace('\n', '').replace('\r', '')

        print(f"[BASE] Evaluating on {clean_dataset}...")

        cmd = [
            sys.executable, "evaluate_models.py",
            "--dataset", clean_dataset,
            "--model_checkpoint", self.config["base_checkpoint"],
            "--config_path", self.config["config_path"],
            "--prompt", self.config["prompt"],
            "--root_dir", self.config["root_dir"]
        ]

        try:
            # Set environment for encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                encoding='utf-8',
                errors='replace'
            )

            # Check if we got results regardless of return code
            accuracy = self.parse_accuracy_from_output(result.stdout)

            # Consider success if we found accuracy values or if stdout contains results
            has_results = (accuracy['top1'] > 0 or accuracy['top5'] > 0 or
                           'top_1_accuracy' in result.stdout or 'accuracy' in result.stdout.lower())

            if has_results:
                print(f"[BASE] Success for {clean_dataset}")

                # Save individual result with safe filename
                result_file = self.batch_results_dir / f"base_{clean_dataset}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Dataset: {clean_dataset}\n")
                    f.write(f"Model: Base MobileCLIP\n")
                    f.write(f"Top-1 Accuracy: {accuracy['top1']:.2f}%\n")
                    f.write(f"Top-5 Accuracy: {accuracy['top5']:.2f}%\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"\nFull Output:\n{result.stdout}\n")
                    f.write(f"\nStderr:\n{result.stderr}\n")

                return accuracy
            else:
                print(f"[BASE] Failed for {clean_dataset}")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                if result.stdout:
                    print(f"Stdout: {result.stdout[:200]}...")
                return {"top1": 0.0, "top5": 0.0}

        except subprocess.TimeoutExpired:
            print(f"[BASE] Timeout for {clean_dataset}")
            return {"top1": 0.0, "top5": 0.0}
        except Exception as e:
            print(f"[BASE] Error for {clean_dataset}: {e}")
            return {"top1": 0.0, "top5": 0.0}

    def run_student_model_evaluation(self, dataset):
        """Run evaluation for KD student model"""
        # Clean dataset name for safe file operations
        clean_dataset = dataset.strip().lower().replace('\n', '').replace('\r', '')

        print(f"[STUDENT] Evaluating on {clean_dataset}...")

        cmd = [
            sys.executable, "evaluate_models_kd.py",
            "--dataset", clean_dataset,
            "--model_checkpoint", self.config["student_checkpoint"],
            "--config_path", self.config["config_path"],
            "--prompt", self.config["prompt"],
            "--root_dir", self.config["root_dir"]
        ]

        try:
            # Set environment for encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                encoding='utf-8',
                errors='replace'
            )

            # Check if we got results regardless of return code
            accuracy = self.parse_accuracy_from_output(result.stdout)

            # Consider success if we found accuracy values or if stdout contains results
            has_results = (accuracy['top1'] > 0 or accuracy['top5'] > 0 or
                           'top_1_accuracy' in result.stdout or 'accuracy' in result.stdout.lower())

            if has_results:
                print(f"[STUDENT] Success for {clean_dataset}")

                # Save individual result with safe filename
                result_file = self.batch_results_dir / f"student_{clean_dataset}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Dataset: {clean_dataset}\n")
                    f.write(f"Model: KD Student MobileCLIP\n")
                    f.write(f"Top-1 Accuracy: {accuracy['top1']:.2f}%\n")
                    f.write(f"Top-5 Accuracy: {accuracy['top5']:.2f}%\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"\nFull Output:\n{result.stdout}\n")
                    f.write(f"\nStderr:\n{result.stderr}\n")

                return accuracy
            else:
                print(f"[STUDENT] Failed for {clean_dataset}")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                if result.stdout:
                    print(f"Stdout: {result.stdout[:200]}...")
                return {"top1": 0.0, "top5": 0.0}

        except subprocess.TimeoutExpired:
            print(f"[STUDENT] Timeout for {clean_dataset}")
            return {"top1": 0.0, "top5": 0.0}
        except Exception as e:
            print(f"[STUDENT] Error for {clean_dataset}: {e}")
            return {"top1": 0.0, "top5": 0.0}

    def run_teacher_model_evaluation(self, dataset):
        """Run evaluation for teacher CLIP model"""
        # Clean dataset name for safe file operations
        clean_dataset = dataset.strip().lower().replace('\n', '').replace('\r', '')

        print(f"[TEACHER] Evaluating on {clean_dataset}...")

        cmd = [
            sys.executable, "evaluate_clip_kd.py",
            "--dataset", clean_dataset,
            "--prompt", self.config["prompt"],
            "--root_dir", self.config["root_dir"]
        ]

        try:
            # Set environment for encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                encoding='utf-8',
                errors='replace'
            )

            # Check if we got results regardless of return code
            accuracy = self.parse_accuracy_from_output(result.stdout)

            # Consider success if we found accuracy values or if stdout contains results
            has_results = (accuracy['top1'] > 0 or accuracy['top5'] > 0 or
                           'top_1_accuracy' in result.stdout or 'accuracy' in result.stdout.lower())

            if has_results:
                print(f"[TEACHER] Success for {clean_dataset}")

                # Save individual result with safe filename
                result_file = self.batch_results_dir / f"teacher_{clean_dataset}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Dataset: {clean_dataset}\n")
                    f.write(f"Model: Teacher CLIP\n")
                    f.write(f"Top-1 Accuracy: {accuracy['top1']:.2f}%\n")
                    f.write(f"Top-5 Accuracy: {accuracy['top5']:.2f}%\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"\nFull Output:\n{result.stdout}\n")
                    f.write(f"\nStderr:\n{result.stderr}\n")

                return accuracy
            else:
                print(f"[TEACHER] Failed for {clean_dataset}")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                if result.stdout:
                    print(f"Stdout: {result.stdout[:200]}...")
                return {"top1": 0.0, "top5": 0.0}

        except subprocess.TimeoutExpired:
            print(f"[TEACHER] Timeout for {clean_dataset}")
            return {"top1": 0.0, "top5": 0.0}
        except Exception as e:
            print(f"[TEACHER] Error for {clean_dataset}: {e}")
            return {"top1": 0.0, "top5": 0.0}

    def run_evaluation_for_dataset(self, dataset):
        """Run evaluation for all three models on a single dataset"""
        # Clean dataset name for file operations
        clean_dataset = dataset.strip().lower().replace('\n', '').replace('\r', '')

        print(f"\n{'=' * 60}")
        print(f"Starting evaluation for dataset: {clean_dataset.upper()}")
        print(f"{'=' * 60}")

        dataset_results = {}

        # Evaluate base model
        base_results = self.run_base_model_evaluation(clean_dataset)
        dataset_results["base"] = base_results

        # Small delay between evaluations
        time.sleep(2)

        # Evaluate student model
        student_results = self.run_student_model_evaluation(clean_dataset)
        dataset_results["student"] = student_results

        # Small delay between evaluations
        time.sleep(2)

        # Evaluate teacher model
        teacher_results = self.run_teacher_model_evaluation(clean_dataset)
        dataset_results["teacher"] = teacher_results

        return dataset_results

    def generate_comparison_table(self):
        """Generate comprehensive comparison table"""
        print(f"\n{'=' * 80}")
        print("GENERATING COMPARISON TABLE")
        print(f"{'=' * 80}")

        # Prepare data for DataFrame
        table_data = []

        for dataset, results in self.results.items():
            # Base model row
            table_data.append({
                "Dataset": dataset.upper(),
                "Model": "MobileCLIP (Base)",
                "Top-1 Accuracy (%)": f"{results['base']['top1']:.2f}",
                "Top-5 Accuracy (%)": f"{results['base']['top5']:.2f}"
            })

            # Student model row
            improvement_top1 = results['student']['top1'] - results['base']['top1']
            improvement_top5 = results['student']['top5'] - results['base']['top5']

            table_data.append({
                "Dataset": "",
                "Model": f"MobileCLIP (KD Student) [+{improvement_top1:.2f}]",
                "Top-1 Accuracy (%)": f"{results['student']['top1']:.2f}",
                "Top-5 Accuracy (%)": f"{results['student']['top5']:.2f}"
            })

            # Teacher model row
            table_data.append({
                "Dataset": "",
                "Model": "CLIP (Teacher)",
                "Top-1 Accuracy (%)": f"{results['teacher']['top1']:.2f}",
                "Top-5 Accuracy (%)": f"{results['teacher']['top5']:.2f}"
            })

            # Add separator
            table_data.append({
                "Dataset": "",
                "Model": "-" * 40,
                "Top-1 Accuracy (%)": "-" * 15,
                "Top-5 Accuracy (%)": "-" * 15
            })

        # Remove last separator
        if table_data:
            table_data.pop()

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Display table
        print("\nRESULTS SUMMARY:")
        print(df.to_string(index=False, justify='left'))

        # Save to files
        csv_file = self.batch_results_dir / "comparison_table.csv"
        json_file = self.batch_results_dir / "comparison_results.json"

        df.to_csv(csv_file, index=False)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")

        # Generate improvement analysis
        self.generate_improvement_analysis()

    def generate_improvement_analysis(self):
        """Generate detailed improvement analysis"""
        print(f"\n{'=' * 80}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'=' * 80}")

        analysis_data = []
        total_improvement_top1 = 0
        total_improvement_top5 = 0
        valid_datasets = 0

        for dataset, results in self.results.items():
            base_top1 = results['base']['top1']
            student_top1 = results['student']['top1']
            teacher_top1 = results['teacher']['top1']

            improvement_top1 = student_top1 - base_top1
            improvement_top5 = results['student']['top5'] - results['base']['top5']

            # Calculate knowledge transfer rate
            if teacher_top1 > base_top1 and teacher_top1 > 0:
                transfer_rate = (improvement_top1 / (teacher_top1 - base_top1)) * 100
            else:
                transfer_rate = 0

            analysis_data.append({
                "Dataset": dataset.upper(),
                "Base -> Student Improvement": f"+{improvement_top1:.2f}%",
                "Knowledge Transfer Rate": f"{transfer_rate:.1f}%",
                "Gap to Teacher": f"{teacher_top1 - student_top1:.2f}%"
            })

            if improvement_top1 != 0:  # Count all non-zero improvements
                total_improvement_top1 += improvement_top1
                total_improvement_top5 += improvement_top5
                valid_datasets += 1

        # Display improvement analysis
        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            print("\nIMPROVEMENT BREAKDOWN:")
            print(analysis_df.to_string(index=False))

        # Summary statistics
        if valid_datasets > 0:
            avg_improvement_top1 = total_improvement_top1 / valid_datasets
            avg_improvement_top5 = total_improvement_top5 / valid_datasets

            print(f"\nSUMMARY STATISTICS:")
            print(f"   Average Top-1 Improvement: +{avg_improvement_top1:.2f}%")
            print(f"   Average Top-5 Improvement: +{avg_improvement_top5:.2f}%")
            print(f"   Datasets with Results: {valid_datasets}/{len(self.results)}")

        # Save analysis
        if analysis_data:
            analysis_file = self.batch_results_dir / "improvement_analysis.csv"
            analysis_df.to_csv(analysis_file, index=False)
            print(f"   Analysis saved to: {analysis_file}")

    def run_batch_evaluation(self, datasets):
        """Run evaluation on all datasets for all models"""
        print(f"Starting batch evaluation for {len(datasets)} datasets")
        print(f"Results directory: {self.batch_results_dir}")

        start_time = time.time()

        for i, dataset in enumerate(datasets, 1):
            # Clean dataset name to avoid issues
            clean_dataset = dataset.strip().lower().replace('\n', '').replace('\r', '')

            print(f"\nProgress: {i}/{len(datasets)} datasets")

            try:
                dataset_results = self.run_evaluation_for_dataset(clean_dataset)
                self.results[clean_dataset] = dataset_results

                print(f"Completed {clean_dataset}: "
                      f"Base={dataset_results['base']['top1']:.1f}%, "
                      f"Student={dataset_results['student']['top1']:.1f}%, "
                      f"Teacher={dataset_results['teacher']['top1']:.1f}%")

            except Exception as e:
                print(f"Failed to evaluate {clean_dataset}: {e}")
                # Continue with next dataset
                continue

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nBatch evaluation completed in {duration / 60:.1f} minutes")

        # Generate final comparison table
        if self.results:
            self.generate_comparison_table()
        else:
            print("No successful evaluations to report")


def main():
    parser = argparse.ArgumentParser(description="Fixed batch evaluation for MobileCLIP KD models")

    parser.add_argument("--base_checkpoint", "-b", required=True,
                        help="Path to base MobileCLIP checkpoint")
    parser.add_argument("--student_checkpoint", "-s", required=True,
                        help="Path to KD student checkpoint")
    parser.add_argument("--config_path", "-c", required=True,
                        help="Path to config file")
    parser.add_argument("--datasets", "-d", nargs='+',
                        default=["cifar10", "cifar100", "food101", "fgcv_aircraft", "caltech_256", "oxford_pets"],
                        help="List of datasets to evaluate. Supported: cifar10, cifar100, food101, fgcv_aircraft, caltech_256, oxford_pets")
    parser.add_argument("--prompt", "-p", default="a photo of a",
                        help="Prompt template")
    parser.add_argument("--root_dir", "-r", default="./eval_datasets",
                        help="Root directory for datasets")

    args = parser.parse_args()

    # Validate dataset names
    supported_datasets = ["cifar10", "cifar100", "food101", "fgcv_aircraft", "caltech_256", "oxford_pets"]
    invalid_datasets = [d for d in args.datasets if d not in supported_datasets]

    if invalid_datasets:
        print(f"Error: Unsupported datasets: {invalid_datasets}")
        print(f"Supported datasets: {supported_datasets}")
        return

    # Configuration
    config = {
        "base_checkpoint": args.base_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "config_path": args.config_path,
        "prompt": args.prompt,
        "root_dir": args.root_dir
    }

    # Validate file paths
    required_files = [config["base_checkpoint"], config["student_checkpoint"], config["config_path"]]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

    print("CONFIGURATION:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    print(f"\nDATASETS TO EVALUATE: {', '.join(args.datasets)}")
    print(f"Supported dataset names: {', '.join(supported_datasets)}")

    # Run batch evaluation
    evaluator = FixedBatchEvaluator(config)
    evaluator.run_batch_evaluation(args.datasets)


if __name__ == "__main__":
    main()