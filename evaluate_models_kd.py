"""
@author: Adityam Ghosh
Date: 10-30-2023
Updated for KD model evaluation

"""

from typing import Callable, List, Dict, Any, Tuple
import numpy as np

import torch

import torch.nn.functional as F
import torch.utils.data as td

import torchvision
import os

import argparse
import yaml
import pickle
import json

from sklearn.metrics import top_k_accuracy_score

from transformers import CLIPTokenizerFast
from utility.transform_data import IMAGENET_COLOR_MEAN, IMAGENET_COLOR_STD








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


def load_student_model_from_kd_checkpoint(checkpoint_path: str, config: Dict):
    """Load student model from KD checkpoint without Lightning wrapper"""
    print(f"Loading student model from KD checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Extract student model weights
    student_weights = {}
    for key, value in state_dict.items():
        if key.startswith('clip_model.student.'):
            new_key = key.replace('clip_model.student.', '')
            student_weights[new_key] = value

    if not student_weights:
        raise ValueError("No student model parameters found in checkpoint")

    print(f"Found {len(student_weights)} student parameters")

    # DEBUG: Check config dimensions before creating model
    print(f"\n🔍 DEBUG CONFIG DIMENSIONS:")
    print(f"  image_model output_dim: {config.get('image_model', {}).get('output_dim', 'NOT_FOUND')}")
    print(f"  text_model output_dim: {config.get('text_model', {}).get('output_dim', 'NOT_FOUND')}")
    print(f"  clip_model proj_dim: {config.get('clip_model', {}).get('proj_dim', 'NOT_FOUND')}")

    # Check for potential mismatches
    img_dim = config.get('image_model', {}).get('output_dim', 0)
    txt_dim = config.get('text_model', {}).get('output_dim', 0)
    proj_dim = config.get('clip_model', {}).get('proj_dim', 0)

    print(f"\n📊 MIProjection will be initialized with:")
    print(f"  Image projection: inp_dim={img_dim}, proj_dim={proj_dim}")
    print(f"  Text projection: inp_dim={txt_dim}, proj_dim={proj_dim}")

    if img_dim == proj_dim == 512:
        print(f"⚠️ POTENTIAL ISSUE: img_dim == proj_dim == 512")
    if txt_dim == proj_dim == 512:
        print(f"⚠️ POTENTIAL ISSUE: txt_dim == proj_dim == 512")

    print(f"=" * 50)

    # Create and load student model
    from models.mobile_clip import MobileCLiP
    student_model = MobileCLiP(config)

    missing, unexpected = student_model.load_state_dict(student_weights, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {len(missing)} parameters")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)} parameters")

    student_model.eval()
    return student_model


def text_process(txt: str, tokenizer: Callable, max_length: int) -> torch.Tensor:
    """
    Function to obtain the text captions as a torch Tensor
    """
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"
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


def get_text_tensors(text_captions: List, model: Callable) -> torch.Tensor:
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
            txt = text_captions[i]["input_ids"].to("cuda:0")
            attn_mask = text_captions[i]["attention_mask"].float().to("cuda:0")
            text_tensors.append(
                F.normalize(model.encode_text(txt, attn_mask), p=2, dim=-1)
                .detach()
                .cpu()
            )

    concat_text_tensor = torch.cat(text_tensors, dim=0)
    return concat_text_tensor


def evaluate(dataloader: Any, model: Callable, text_tensors: torch.Tensor) -> Dict:
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

    true_labels = list()
    pred_labels = list()
    preds = list()

    with prog_bar as p:
        model.eval()
        for batch in p.track(dataloader, description="Evaluating Model"):
            img = batch[0]
            label = batch[1]

            img_encoding = model.encode_image(img.to("cuda:0"))
            img_encoding = F.normalize(img_encoding, p=2, dim=-1)
            text_tensors = text_tensors.cuda()
            similarities = (100.0 * img_encoding @ text_tensors.t()).softmax(dim=-1)
            pred_label = torch.argmax(similarities, dim=1)

            true_labels.append(label.detach().numpy())
            preds.append(similarities.detach().cpu().numpy())
            pred_labels.append(pred_label.detach().cpu().numpy())

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    preds = np.concatenate(preds, axis=0)



    unique_preds, counts = np.unique(pred_labels, return_counts=True)
    unique_preds = unique_preds.reshape(-1, 1)
    counts = counts.reshape(-1, 1)
    print(f"Frequency of each predicted class by the model:")
    print(np.hstack((unique_preds, counts)))

    return {
        "top_1_accuracy": top_k_accuracy_score(true_labels, preds, k=1),
        "top_5_accuracy": top_k_accuracy_score(true_labels, preds, k=5),
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
    cfg: Dict,
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
        text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
        for txt in prompts
    ]

    txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
    test_dl = td.DataLoader(fgcv_dataset, batch_size=1, shuffle=False, num_workers=2)
    acc = evaluate(test_dl, model, txt_tensors)

    print(acc)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        prog="evaluation script",
        description="Evaluation script for the Mobile CLiP model with KD",
    )

    parser.add_argument(
        "--root_dir",
        "-r",
        required=False,
        type=str,
        default="./eval_datasets",
        help="the location where to download the datasets if not present",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=str,
        help="the name of the dataset for evaluation",
    )

    parser.add_argument(
        "--model_checkpoint",
        "-c",
        required=True,
        type=str,
        help="the model checkpoint location",
    )

    parser.add_argument(
        "--config_path",
        "-p",
        required=True,
        type=str,
        help="the config path for the models",
    )

    parser.add_argument(
        "--prompt",
        "-P",
        required=False,
        type=str,
        default="a photo of a",
        help="the prompt template to use",
    )

    parser.add_argument(
        "--save_results",
        "-s",
        required=False,
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        required=False,
        type=int,
        default=1,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_name = args.dataset
    model_checkpoint = args.model_checkpoint
    config_path = args.config_path
    prompt_template = args.prompt.strip()
    save_path = args.save_results
    batch_size = args.batch_size

    cfg = None
    with open(config_path, "r") as fp:
        try:
            cfg = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    # FIXED: Load student model directly from KD checkpoint

    model = load_student_model_from_kd_checkpoint(model_checkpoint, cfg)

    # Move to GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    text_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD
            ),
        ]
    )

    # Store results
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
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        print("\nClass labels:")
        pprint(label_map)

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)

        test_dl = td.DataLoader(
            cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors)
        print("\nResults on CIFAR-10:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")

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
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        print("\nFirst 10 class labels:")
        pprint({k: v for i, (k, v) in enumerate(label_map.items()) if i < 10})
        print(f"... (total: {len(label_map)} classes)")

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)

        test_dl = td.DataLoader(
            cifar100_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors)
        print("\nResults on CIFAR-100:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")

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
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        print("\nFirst 10 class labels:")
        pprint({k: v for i, (k, v) in enumerate(label_map.items()) if i < 10})
        print(f"... (total: {len(label_map)} classes)")

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            food101_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors)
        print("\nResults on Food-101:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")

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
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="family",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_families,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )

        print("\n########## Annotation Level : Variant ############")
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="variant",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_variant,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )

        print("\n########## Annotation Level : Manufacturer ############")
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="manufacturer",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_manufactures,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )



    elif dataset_name == "caltech_256":
        print(f"\n=== Evaluating on Caltech-256 dataset ===")


        class RepeatChannels:
            def __call__(self, tensor):
                if tensor.shape[0] == 1:
                    return tensor.repeat(3, 1, 1)
                return tensor


        transformations_caltech = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            RepeatChannels(),  # ✅ FIXED
            torchvision.transforms.Normalize(
                mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD
            ),
        ])


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
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            caltech_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        results = evaluate(test_dl, model, txt_tensors)
        print("\nResults on Caltech-256:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")

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
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            oxford_pets_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        results = evaluate(test_dl, model, txt_tensors)
        print("\nResults on Oxford-IIIT Pets:")
        print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")

    # Save results if requested
    if save_path and results:
        results_to_save = {
            "dataset": dataset_name,
            "model": os.path.basename(model_checkpoint),
            "prompt": prompt_template,
            "top_1_accuracy": float(results.get("top_1_accuracy", 0)),
            "top_5_accuracy": float(results.get("top_5_accuracy", 0)),
            "batch_size": batch_size
        }

        print(f"\nSaving results to {save_path}")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print("✓ Results saved successfully")