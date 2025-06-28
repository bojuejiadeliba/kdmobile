"""
@author: Evaluation script for OpenAI CLIP (Teacher) model
Date: 2025-05-13

This script evaluates the OpenAI CLIP model on zero-shot image classification tasks,
using the same format and output style as evaluate_models_kd.py for consistency.
"""

from typing import Callable, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision
import os
import sys
import argparse
import yaml
import pickle
import json

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from transformers import CLIPTokenizerFast, CLIPModel, CLIPProcessor
from PIL import Image
from utility.transform_data import (
    NormalizeCaption,
    IMAGENET_COLOR_MEAN,
    IMAGENET_COLOR_STD,
)
import albumentations as alb

from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
    BarColumn,
)
from pprint import pprint

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import sys
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def text_process(txt: str, tokenizer: Callable, max_length: int) -> torch.Tensor:
    """
    Function to obtain the text captions as a torch Tensor
    """
    txt = txt.lower()
    txt += "."

    tok_outputs = tokenizer(
        txt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tok_outputs


def get_text_tensors(text_captions: List, model: Callable, device: str = "cuda:0") -> torch.Tensor:
    """Function to obtain the text tensors for all the captions"""

    prog_bar = Progress(
        TextColumn("[progress.percentage] {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    text_tensors = []

    with prog_bar as p:
        n_captions = len(text_captions)

        for i in p.track(range(n_captions), description="Getting text tensors"):
            txt = text_captions[i]["input_ids"].to(device)
            attn_mask = text_captions[i]["attention_mask"].float().to(device)

            with torch.no_grad():
                # Use CLIP's text feature extraction
                text_features = model.get_text_features(
                    input_ids=txt,
                    attention_mask=attn_mask
                )
                # Normalize features
                normalized_features = F.normalize(text_features, p=2, dim=-1)
                text_tensors.append(normalized_features.detach().cpu())

    concat_text_tensor = torch.cat(text_tensors, dim=0)
    return concat_text_tensor


def evaluate(dataloader: Any, model: Callable, text_tensors: torch.Tensor,
           device: str = "cuda:0", processor: Any = None) -> Dict:
    """Function to evaluate the model"""

    prog_bar = Progress(
        TextColumn("[progress.percentage] {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    true_labels = []
    pred_labels = []
    preds = []
    batch_times = []

    import time

    with prog_bar as p:
        model.eval()
        for batch in p.track(dataloader, description="Evaluating Model"):
            start_time = time.time()

            img, label = batch

            # For CLIP, we need to use the processor to prepare images
            # Disable rescaling since images are already normalized
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()
            processed = processor(images=img_np, return_tensors="pt", do_rescale=False)
            processed = {k: v.to(device) for k, v in processed.items()}

            with torch.no_grad():
                img_encoding = model.get_image_features(**processed)
                img_encoding = F.normalize(img_encoding, p=2, dim=-1)

                text_tensors_gpu = text_tensors.to(device)
                similarities = (100.0 * img_encoding @ text_tensors_gpu.t()).softmax(dim=-1)
                pred_label = torch.argmax(similarities, dim=1)

                # Convert to numpy and handle batch dimension properly
                label_np = label.detach().cpu().numpy()  # This is a batch of labels
                true_labels.extend(label_np)  # Extend with individual labels

                preds.append(similarities.detach().cpu().numpy())
                pred_labels.extend(pred_label.detach().cpu().numpy())

            batch_time = time.time() - start_time
            batch_times.append(batch_time)

    # Concatenate predictions from all batches
    preds = np.concatenate(preds, axis=0)

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    print(f"Predictions shape: {preds.shape}")
    print(f"True labels shape: {true_labels.shape}")
    print(f"Pred labels shape: {pred_labels.shape}")

    unique_preds, counts = np.unique(pred_labels, return_counts=True)
    unique_preds = unique_preds.reshape(-1, 1)
    counts = counts.reshape(-1, 1)
    print(f"Frequency of each predicted class by the model:")
    print(np.hstack((unique_preds, counts)))

    avg_time = sum(batch_times) / len(batch_times)
    print(f"Average inference time per batch: {avg_time:.4f}s")

    return {
        "top_1_accuracy": top_k_accuracy_score(true_labels, preds, k=1),
        "top_5_accuracy": top_k_accuracy_score(true_labels, preds, k=5),
        "avg_inference_time": avg_time
    }


def unpickle(file_path: str) -> Dict:
    """Function to unpickle the meta data of the CIFAR dataset"""

    with open(file_path, "rb") as fp:
        data = pickle.load(fp)

    return data


def fgvc_aircraft_res(
    root_dir: str,
    annotation_level: str,
    transformations: Callable,
    text_tokenizer: Callable,
    meta_path: str,
    prompt_template: str,
    model: Callable,
    processor: Callable,
    device: str,
    max_seq_length: int = 77,
):
    """Function to obtain results for the FGVC Aircraft dataset"""

    fgcv_dataset = torchvision.datasets.FGVCAircraft(
        root=root_dir,
        split="test",
        annotation_level=annotation_level,
        transform=transformations,
        download=True,
    )

    with open(meta_path, "r") as fp:
        meta_data = fp.readlines()

    label_map = {k.replace("\n", ""): idx for idx, k in enumerate(meta_data)}
    pprint(label_map)

    prompts = list(
        map(
            lambda x: prompt_template + " " + x.replace("\n", ""),
            meta_data,
        )
    )

    tokenized_txts = [
        text_process(txt, text_tokenizer, max_seq_length)
        for txt in prompts
    ]

    txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)
    test_dl = td.DataLoader(fgcv_dataset, batch_size=32, shuffle=False, num_workers=2)
    acc = evaluate(test_dl, model, txt_tensors, device, processor)

    print(acc)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        prog="evaluation script",
        description="Evaluation script for the OpenAI CLIP Teacher model",
    )

    parser.add_argument(
        "--root_dir",
        "-r",
        required=False,
        type=str,
        default="./eval_datasets",
        help="The location where to download the datasets if not present",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=str,
        help="The name of the dataset for evaluation",
    )

    parser.add_argument(
        "--model_name",
        "-m",
        required=False,
        type=str,
        default="openai/clip-vit-base-patch32",
        help="The OpenAI CLIP model name to use",
    )

    parser.add_argument(
        "--prompt",
        "-P",
        required=False,
        type=str,
        default="a photo of a",
        help="The prompt template to use",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--save_results",
        "-s",
        required=False,
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_name = args.dataset
    model_name = args.model_name
    prompt_template = args.prompt.strip()
    batch_size = args.batch_size
    save_path = args.save_results

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the CLIP model
    print(f"Loading OpenAI CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    text_tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    # Standard transformations for initial loading - no normalization as CLIP processor handles it
    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    # Max sequence length is standard 77 for CLIP
    max_seq_length = 77

    # Dictionary to store results
    results = {}

    if dataset_name == "cifar10":
        print(f"\n=== Evaluating on CIFAR-10 dataset ===")
        cifar10_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transformations
        )

        meta_path = os.path.join(root_dir, "cifar-10-batches-py/batches.meta")
        meta_data = unpickle(meta_path)
        prompts = list(
            map(lambda x: prompt_template + " " + x, meta_data["label_names"])
        )
        label_map = {k: idx for idx, k in enumerate(meta_data["label_names"])}

        tokenized_txts = [
            text_process(txt, text_tokenizer, max_seq_length)
            for txt in prompts
        ]

        print("\nClass labels:")
        pprint(label_map)

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)

        test_dl = td.DataLoader(
            cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors, device, processor)
        print("\nResults on CIFAR-10:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")

    elif dataset_name == "cifar100":
        print(f"\n=== Evaluating on CIFAR-100 dataset ===")
        cifar100_dataset = torchvision.datasets.CIFAR100(
            root=root_dir, train=False, download=True, transform=transformations
        )

        meta_path = os.path.join(root_dir, "cifar-100-python/meta")
        meta_data = unpickle(meta_path)
        prompts = list(
            map(lambda x: prompt_template + " " + x, meta_data["fine_label_names"])
        )
        label_map = {k: idx for idx, k in enumerate(meta_data["fine_label_names"])}
        tokenized_txts = [
            text_process(txt, text_tokenizer, max_seq_length)
            for txt in prompts
        ]

        print("\nFirst 10 class labels:")
        pprint({k: v for i, (k, v) in enumerate(label_map.items()) if i < 10})
        print(f"... (total: {len(label_map)} classes)")

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)

        test_dl = td.DataLoader(
            cifar100_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors, device, processor)
        print("\nResults on CIFAR-100:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")

    elif dataset_name == "food101":
        print(f"\n=== Evaluating on Food-101 dataset ===")
        food101_dataset = torchvision.datasets.Food101(
            root=root_dir, split="test", transform=transformations, download=True
        )

        meta_path = os.path.join(root_dir, "food-101/meta/labels.txt")
        with open(meta_path, "r") as fp:
            meta_data = fp.readlines()

        label_map = {k.replace("\n", ""): idx for idx, k in enumerate(meta_data)}
        prompts = list(
            map(lambda x: prompt_template + " " + x.replace("\n", ""), meta_data)
        )
        tokenized_txts = [
            text_process(txt, text_tokenizer, max_seq_length)
            for txt in prompts
        ]

        print("\nFirst 10 class labels:")
        pprint({k: v for i, (k, v) in enumerate(label_map.items()) if i < 10})
        print(f"... (total: {len(label_map)} classes)")

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)
        test_dl = td.DataLoader(
            food101_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors, device, processor)
        print("\nResults on Food-101:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")

    elif dataset_name == "fgcv_aircraft":
        print(f"\n=== Evaluating on FGVC Aircraft dataset ===")
        meta_path_families = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/families.txt"
        )
        meta_path_variant = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/variants.txt"
        )
        meta_path_manufactures = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/manufacturers.txt"
        )

        print("\n########## Annotation Level : Family ############")
        family_results = fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="family",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_families,
            prompt_template=prompt_template,
            model=model,
            processor=processor,
            device=device,
            max_seq_length=max_seq_length,
        )

        print("\n########## Annotation Level : Variant ############")
        variant_results = fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="variant",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_variant,
            prompt_template=prompt_template,
            model=model,
            processor=processor,
            device=device,
            max_seq_length=max_seq_length,
        )

        print("\n########## Annotation Level : Manufacturer ############")
        manufacturer_results = fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="manufacturer",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_manufactures,
            prompt_template=prompt_template,
            model=model,
            processor=processor,
            device=device,
            max_seq_length=max_seq_length,
        )

    elif dataset_name == "caltech_256":
        print(f"\n=== Evaluating on Caltech-256 dataset ===")
        class RepeatChannels:
            def __call__(self, tensor):
                if tensor.shape[0] == 1:
                    return tensor.repeat(3, 1, 1)
                return tensor
        transformations_caltech = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                #RepeatChannels(),
                torchvision.transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                ),
                torchvision.transforms.Normalize(
                    mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD
                ),
            ]
        )
        caltech_dataset = torchvision.datasets.Caltech256(
            root=root_dir, transform=transformations_caltech, download=False
        )

        meta_path = os.path.join(root_dir, "caltech256/caltech_labels.json")

        with open(meta_path, "r") as fp:
            meta_data = json.load(fp)

        print("\nFirst 10 class labels:")
        pprint({k: v for i, (k, v) in enumerate(meta_data.items()) if i < 10})
        print(f"... (total: {len(meta_data)} classes)")

        prompts = [prompt_template + " " + labels for labels, _ in meta_data.items()]

        tokenized_txts = [
            text_process(txt, text_tokenizer, max_seq_length)
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)
        test_dl = td.DataLoader(
            caltech_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        results = evaluate(test_dl, model, txt_tensors, device, processor)
        print("\nResults on Caltech-256:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")

    elif dataset_name == "oxford_pets":
        print(f"\n=== Evaluating on Oxford-IIIT Pets dataset ===")
        oxford_pets_dataset = torchvision.datasets.OxfordIIITPet(
            root=root_dir,
            split="test",
            transform=transformations,
            target_types="category",
            download=True,
        )

        meta_path = os.path.join(root_dir, "oxford-iiit-pet/oxford_pets_labels.json")

        with open(meta_path, "r") as fp:
            meta_data = json.load(fp)

        print("\nClass labels:")
        pprint(dict(sorted(meta_data.items(), key=lambda x: x[1])))

        prompts = [prompt_template + " " + labels for labels, _ in meta_data.items()]

        tokenized_txts = [
            text_process(txt, text_tokenizer, max_seq_length)
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model, device=device)
        test_dl = td.DataLoader(
            oxford_pets_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors, device, processor)
        print("\nResults on Oxford-IIIT Pets:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"Average inference time: {results['avg_inference_time']:.4f}s")

    # Save results if path provided
    if save_path:
        results_to_save = {
            "dataset": dataset_name,
            "model": model_name,
            "prompt": prompt_template,
            "top_1_accuracy": float(results.get("top_1_accuracy", 0)),
            "top_5_accuracy": float(results.get("top_5_accuracy", 0)),
            "avg_inference_time": float(results.get("avg_inference_time", 0)),
            "batch_size": batch_size
        }

        print(f"\nSaving results to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print("✓ Results saved successfully")