#!/usr/bin/env python3
"""
Model Efficiency Analysis for Mobile Deployment
Analyzes parameter count, model size, memory usage, and inference speed
"""

import torch
import torch.nn as nn
import os
import time
import psutil
import yaml
import argparse
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np

# Import your models
from models.mobile_clip import MobileCLiP
from models.mobile_clip_kd import MobileCLIPWithKD
from trainer_kd import LitMobileCLiPKD
from trainer_auto_kd import LitMobileCLiPAutoKD


class ModelEfficiencyAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "total_params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6
        }

    def get_model_size(self, model_path):
        """Get saved model file size"""
        if not os.path.exists(model_path):
            return {"error": "File not found"}

        file_size = os.path.getsize(model_path)
        return {
            "file_size_bytes": file_size,
            "file_size_MB": file_size / (1024 * 1024),
            "file_size_GB": file_size / (1024 * 1024 * 1024)
        }

    def estimate_memory_usage(self, model, input_shape=(1, 3, 224, 224), text_length=77):
        """Estimate memory usage during inference"""
        model.eval()
        model = model.to(self.device)

        # Create dummy inputs
        dummy_image = torch.randn(input_shape).to(self.device)
        dummy_text = torch.randint(0, 49408, (input_shape[0], text_length)).to(self.device)
        dummy_mask = torch.ones(input_shape[0], text_length).to(self.device)

        # Measure memory before
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()

        # Forward pass
        with torch.no_grad():
            try:
                if hasattr(model, 'clip_model'):  # KD models
                    _ = model.clip_model.student(dummy_image, dummy_text, dummy_mask)
                elif hasattr(model, 'vision_model'):  # CLIP teacher model
                    # CLIP model forward pass
                    _ = model.get_image_features(dummy_image)
                    _ = model.get_text_features(dummy_text)
                else:  # Regular MobileCLIP models
                    _ = model(dummy_image, dummy_text, dummy_mask)
            except Exception as e:
                print(f"Forward pass failed: {e}")
                return {"error": str(e)}

        # Measure memory after
        if self.device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            return {
                "memory_usage_MB": (memory_after - memory_before) / (1024 * 1024),
                "peak_memory_MB": peak_memory / (1024 * 1024),
                "device": "GPU"
            }
        else:
            # For CPU, estimate based on model parameters
            param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # 4 bytes per float32
            return {
                "estimated_memory_MB": param_memory,
                "device": "CPU"
            }

    def measure_inference_speed(self, model, num_runs=100, input_shape=(1, 3, 224, 224), text_length=77):
        """Measure inference speed"""
        model.eval()
        model = model.to(self.device)

        # Create dummy inputs
        dummy_image = torch.randn(input_shape).to(self.device)
        dummy_text = torch.randint(0, 49408, (input_shape[0], text_length)).to(self.device)
        dummy_mask = torch.ones(input_shape[0], text_length).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    if hasattr(model, 'clip_model'):  # KD models
                        _ = model.clip_model.student(dummy_image, dummy_text, dummy_mask)
                    elif hasattr(model, 'vision_model'):  # CLIP teacher model
                        _ = model.get_image_features(dummy_image)
                        _ = model.get_text_features(dummy_text)
                    else:  # Regular MobileCLIP models
                        _ = model(dummy_image, dummy_text, dummy_mask)
                except Exception as e:
                    return {"error": str(e)}

        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    if hasattr(model, 'clip_model'):  # KD models
                        _ = model.clip_model.student(dummy_image, dummy_text, dummy_mask)
                    elif hasattr(model, 'vision_model'):  # CLIP teacher model
                        _ = model.get_image_features(dummy_image)
                        _ = model.get_text_features(dummy_text)
                    else:  # Regular MobileCLIP models
                        _ = model(dummy_image, dummy_text, dummy_mask)
                except Exception as e:
                    return {"error": str(e)}

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                times.append(end_time - start_time)

        times = np.array(times)
        return {
            "avg_inference_time_ms": np.mean(times) * 1000,
            "std_inference_time_ms": np.std(times) * 1000,
            "min_inference_time_ms": np.min(times) * 1000,
            "max_inference_time_ms": np.max(times) * 1000,
            "fps": 1.0 / np.mean(times),
            "num_runs": num_runs
        }

    def load_teacher_model(self, teacher_name="openai/clip-vit-base-patch32"):
        """Load teacher CLIP model"""
        print(f"Loading teacher model: {teacher_name}")

        from transformers import CLIPModel
        teacher_model = CLIPModel.from_pretrained(teacher_name)
        return teacher_model

    def load_model_from_checkpoint(self, checkpoint_path, config_path, model_type="auto"):
        """Load model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if model_type == "teacher":
            # Teacher CLIP model
            print("Loading teacher CLIP model")
            teacher_name = config.get("knowledge_distillation", {}).get("teacher_model", "openai/clip-vit-base-patch32")
            return self.load_teacher_model(teacher_name)

        elif model_type == "base" or "mobile_test" in checkpoint_path:
            # Base MobileCLIP
            print("Loading as base MobileCLIP model")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Create model
            model = MobileCLiP(config)

            # Load weights
            if 'state_dict' in checkpoint:
                # Lightning checkpoint
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('clip_model.'):
                        new_key = key.replace('clip_model.', '')
                        state_dict[new_key] = value
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            return model

        else:
            # KD model - extract student only
            print("Loading as KD student model")

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Create student model
            student_model = MobileCLiP(config)

            # Extract student weights
            if 'state_dict' in checkpoint:
                student_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('clip_model.student.'):
                        new_key = key.replace('clip_model.student.', '')
                        student_state_dict[new_key] = value

                if student_state_dict:
                    student_model.load_state_dict(student_state_dict, strict=False)
                    print(f"Loaded {len(student_state_dict)} student parameters")
                else:
                    print("Warning: No student parameters found in checkpoint")

            return student_model

    def analyze_model(self, checkpoint_path, config_path, model_name, model_type="auto"):
        """Complete analysis of a single model"""
        print(f"\n{'=' * 60}")
        print(f"Analyzing {model_name}")
        print(f"{'=' * 60}")

        analysis = {
            "model_name": model_name,
            "checkpoint_path": checkpoint_path if model_type != "teacher" else "HuggingFace Hub",
            "model_type": model_type
        }

        # File size analysis (skip for teacher model from HuggingFace)
        if model_type != "teacher":
            print("1. Analyzing file size...")
            analysis["file_size"] = self.get_model_size(checkpoint_path)
        else:
            print("1. Teacher model from HuggingFace Hub...")
            analysis["file_size"] = {"note": "Downloaded from HuggingFace Hub", "file_size_MB": "N/A"}

        # Load model
        try:
            if model_type == "teacher":
                model = self.load_model_from_checkpoint(None, config_path, "teacher")
            else:
                model = self.load_model_from_checkpoint(checkpoint_path, config_path, model_type)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            analysis["error"] = str(e)
            return analysis

        # Parameter analysis
        print("2. Counting parameters...")
        analysis["parameters"] = self.count_parameters(model)

        # Memory analysis
        print("3. Measuring memory usage...")
        analysis["memory"] = self.estimate_memory_usage(model)

        # Speed analysis
        print("4. Measuring inference speed...")
        analysis["speed"] = self.measure_inference_speed(model)

        return analysis

    def compare_models(self, analyses):
        """Compare multiple model analyses"""
        print(f"\n{'=' * 80}")
        print("MODEL COMPARISON FOR MOBILE DEPLOYMENT")
        print(f"{'=' * 80}")

        # Create comparison table
        comparison_data = []

        for analysis in analyses:
            if "error" in analysis:
                continue

            file_size = analysis['file_size'].get('file_size_MB', 'N/A')
            if file_size != 'N/A':
                file_size = f"{file_size:.1f}"

            row = {
                "Model": analysis["model_name"],
                "Parameters (M)": f"{analysis['parameters']['total_params_M']:.1f}",
                "File Size (MB)": file_size,
                "Memory (MB)": f"{analysis['memory'].get('peak_memory_MB', analysis['memory'].get('estimated_memory_MB', 0)):.1f}",
                "Inference (ms)": f"{analysis['speed'].get('avg_inference_time_ms', 0):.1f}",
                "FPS": f"{analysis['speed'].get('fps', 0):.1f}"
            }
            comparison_data.append(row)

        # Display comparison
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            print("\nCOMPARISON TABLE:")
            print(df.to_string(index=False))

            # Efficiency gains analysis
            self.efficiency_gains_analysis(analyses)

            # Mobile deployment recommendations
            self.mobile_deployment_analysis(analyses)

        return comparison_data

    def efficiency_gains_analysis(self, analyses):
        """Analyze efficiency gains from knowledge distillation"""
        print(f"\n{'=' * 80}")
        print("KNOWLEDGE DISTILLATION EFFICIENCY GAINS")
        print(f"{'=' * 80}")

        # Find teacher and student models
        teacher_analysis = None
        student_analysis = None
        base_analysis = None

        for analysis in analyses:
            if "error" in analysis:
                continue
            if "Teacher" in analysis["model_name"]:
                teacher_analysis = analysis
            elif "Student" in analysis["model_name"]:
                student_analysis = analysis
            elif "Base" in analysis["model_name"]:
                base_analysis = analysis

        if teacher_analysis and student_analysis:
            print("STUDENT vs TEACHER EFFICIENCY:")

            # Parameter reduction
            teacher_params = teacher_analysis['parameters']['total_params_M']
            student_params = student_analysis['parameters']['total_params_M']
            param_reduction = ((teacher_params - student_params) / teacher_params) * 100

            print(f"  Parameter Reduction: {param_reduction:.1f}% ({teacher_params:.1f}M -> {student_params:.1f}M)")

            # Memory reduction
            teacher_memory = teacher_analysis['memory'].get('peak_memory_MB',
                                                            teacher_analysis['memory'].get('estimated_memory_MB', 0))
            student_memory = student_analysis['memory'].get('peak_memory_MB',
                                                            student_analysis['memory'].get('estimated_memory_MB', 0))
            if teacher_memory > 0 and student_memory > 0:
                memory_reduction = ((teacher_memory - student_memory) / teacher_memory) * 100
                print(f"  Memory Reduction: {memory_reduction:.1f}% ({teacher_memory:.1f}MB -> {student_memory:.1f}MB)")

            # Speed improvement
            teacher_time = teacher_analysis['speed'].get('avg_inference_time_ms', 0)
            student_time = student_analysis['speed'].get('avg_inference_time_ms', 0)
            if teacher_time > 0 and student_time > 0:
                speed_improvement = ((teacher_time - student_time) / teacher_time) * 100
                speedup_factor = teacher_time / student_time
                print(f"  Speed Improvement: {speed_improvement:.1f}% ({teacher_time:.1f}ms -> {student_time:.1f}ms)")
                print(f"  Speedup Factor: {speedup_factor:.1f}x faster")

            # Compression ratio
            compression_ratio = teacher_params / student_params
            print(f"  Model Compression Ratio: {compression_ratio:.1f}x smaller")

        if base_analysis and student_analysis:
            print(f"\nSTUDENT vs BASE (Same Architecture):")
            base_params = base_analysis['parameters']['total_params_M']
            student_params = student_analysis['parameters']['total_params_M']
            print(f"  Parameters: {base_params:.1f}M vs {student_params:.1f}M (should be identical)")
            print(f"  Note: Student model has same efficiency as base, but improved accuracy from KD")

    def mobile_deployment_analysis(self, analyses):
        """Analyze suitability for mobile deployment"""
        print(f"\n{'=' * 80}")
        print("MOBILE DEPLOYMENT SUITABILITY")
        print(f"{'=' * 80}")

        # Mobile deployment thresholds
        thresholds = {
            "excellent": {"params_M": 10, "size_MB": 40, "memory_MB": 100, "inference_ms": 50},
            "good": {"params_M": 25, "size_MB": 100, "memory_MB": 200, "inference_ms": 100},
            "acceptable": {"params_M": 50, "size_MB": 200, "memory_MB": 500, "inference_ms": 200}
        }

        for analysis in analyses:
            if "error" in analysis:
                continue

            model_name = analysis["model_name"]
            params_M = analysis['parameters']['total_params_M']

            # Handle file size safely
            file_size_data = analysis['file_size']
            if isinstance(file_size_data.get('file_size_MB'), str) or file_size_data.get('file_size_MB') == 'N/A':
                size_MB = 0  # For teacher model from HuggingFace
                size_display = "N/A"
            else:
                size_MB = file_size_data.get('file_size_MB', 0)
                size_display = f"{size_MB:.1f}MB"

            memory_MB = analysis['memory'].get('peak_memory_MB', analysis['memory'].get('estimated_memory_MB', 0))
            inference_ms = analysis['speed'].get('avg_inference_time_ms', 0)

            print(f"\n{model_name}:")
            print(f"  Parameters: {params_M:.1f}M")
            print(f"  File Size: {size_display}")
            print(f"  Memory: {memory_MB:.1f}MB")
            print(f"  Inference: {inference_ms:.1f}ms")

            # Determine suitability
            if (params_M <= thresholds["excellent"]["params_M"] and
                    size_MB <= thresholds["excellent"]["size_MB"] and
                    memory_MB <= thresholds["excellent"]["memory_MB"] and
                    inference_ms <= thresholds["excellent"]["inference_ms"]):
                rating = "EXCELLENT for mobile"
                color = "✅"
            elif (params_M <= thresholds["good"]["params_M"] and
                  size_MB <= thresholds["good"]["size_MB"] and
                  memory_MB <= thresholds["good"]["memory_MB"] and
                  inference_ms <= thresholds["good"]["inference_ms"]):
                rating = "GOOD for mobile"
                color = "🟢"
            elif (params_M <= thresholds["acceptable"]["params_M"] and
                  size_MB <= thresholds["acceptable"]["size_MB"] and
                  memory_MB <= thresholds["acceptable"]["memory_MB"] and
                  inference_ms <= thresholds["acceptable"]["inference_ms"]):
                rating = "ACCEPTABLE for mobile"
                color = "🟡"
            else:
                rating = "NOT SUITABLE for mobile"
                color = "🔴"

            print(f"  Mobile Rating: {color} {rating}")

    def save_analysis(self, analyses, output_file="model_analysis.json"):
        """Save analysis to file"""
        with open(output_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        print(f"\nAnalysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model efficiency for mobile deployment")

    parser.add_argument("--base_checkpoint", "-b",
                        help="Path to base MobileCLIP checkpoint")
    parser.add_argument("--student_checkpoint", "-s",
                        help="Path to KD student checkpoint")
    parser.add_argument("--config_path", "-c", required=True,
                        help="Path to config file")
    parser.add_argument("--include_teacher", "-t", action="store_true",
                        help="Include teacher CLIP model analysis")
    parser.add_argument("--output", "-o", default="model_analysis.json",
                        help="Output file for analysis results")

    args = parser.parse_args()

    analyzer = ModelEfficiencyAnalyzer()
    analyses = []

    # Analyze teacher model if requested
    if args.include_teacher:
        teacher_analysis = analyzer.analyze_model(
            None,  # No checkpoint path for teacher
            args.config_path,
            "Teacher CLIP",
            "teacher"
        )
        analyses.append(teacher_analysis)

    # Analyze base model if provided
    if args.base_checkpoint and os.path.exists(args.base_checkpoint):
        base_analysis = analyzer.analyze_model(
            args.base_checkpoint,
            args.config_path,
            "Base MobileCLIP",
            "base"
        )
        analyses.append(base_analysis)

    # Analyze student model if provided
    if args.student_checkpoint and os.path.exists(args.student_checkpoint):
        student_analysis = analyzer.analyze_model(
            args.student_checkpoint,
            args.config_path,
            "KD Student MobileCLIP",
            "student"
        )
        analyses.append(student_analysis)

    if not analyses:
        print("No valid checkpoints provided!")
        return

    # Compare models
    comparison_data = analyzer.compare_models(analyses)

    # Save results
    analyzer.save_analysis(analyses, args.output)


if __name__ == "__main__":
    main()