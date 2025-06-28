"""
@author: Knowledge Distillation Training Script for Mobile CLIP
Date: 2025-05-12

This script trains Mobile CLIP using knowledge distillation from a teacher CLIP model.
"""

import argparse
import numpy as np
import pandas as pd
import polars as pol
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import torchvision
import albumentations as alb
import os
import sys
import yaml
import zipfile
## import utility.amp_patch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import CLIPTokenizerFast, ConvBertTokenizer

from trainer_kd import LitMobileCLiPKD

from trainer_auto_kd import LitMobileCLiPAutoKD, load_automatic_kd_config
from utility.datasets import CocoDataset
from utility.transform_data import (
    HorizontalFlip,
    IMAGENET_COLOR_MEAN,
    IMAGENET_COLOR_STD,
    CenterSquareCrop,
)

# Store original functions for safe gradient clipping
original_clip_grad_value = torch.nn.utils.clip_grad_value_
original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

# Fix gradient clipping functions to handle empty gradients safely
def safe_clip_grad_value_(parameters, clip_value, **kwargs):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters_with_grad = [p for p in parameters if p is not None and p.grad is not None]

    if len(parameters_with_grad) > 0:
        return original_clip_grad_value(parameters_with_grad, clip_value, **kwargs)
    return None


def safe_clip_grad_norm_(parameters, max_norm, norm_type=2.0, **kwargs):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters_with_grad = [p for p in parameters if p is not None and p.grad is not None]

    if len(parameters_with_grad) > 0:
        return original_clip_grad_norm(parameters_with_grad, max_norm, norm_type, **kwargs)
    return 0.0


# Replace the functions
torch.nn.utils.clip_grad_value_ = safe_clip_grad_value_
torch.nn.utils.clip_grad_norm_ = safe_clip_grad_norm_


def debug_first_batch(batch, config):
    """Debug function to check the first batch structure"""
    print("\n=== DEBUG: First Batch Structure ===")
    for key, value in batch.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                else:
                    print(f"  {sub_key}: {type(sub_value)}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {type(value)}")

    print("\n=== Teacher Model Config ===")
    for key, value in config.get("knowledge_distillation", {}).items():
        print(f"{key}: {value}")
    print("============================\n")





if __name__ == "__main__":
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()

    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="mobile_clip_kd_training",
        description="Train Mobile CLIP with Knowledge Distillation"
    )

    # Data paths
    parser.add_argument(
        "--auto_kd_config",
        "-akd",
        required=False,
        type=str,
        default="config_kd.yaml",  # Your automatic KD config file
        help="Path to automatic KD configuration file"
    )
    parser.add_argument(
        "--train_data_path",
        "-P",
        required=True,
        type=str,
        help="CSV file path for the train data",
    )
    parser.add_argument(
        "--val_data_path",
        "-V",
        required=True,
        type=str,
        help="CSV file path for the val data",
    )
    parser.add_argument(
        "--config_path",
        "-p",
        required=True,
        type=str,
        help="Config file path"
    )

    # Training parameters
    parser.add_argument(
        "--max_epochs",
        "-E",
        required=False,
        type=int,
        default=50,
        help="Maximum number of epochs to train the model",
    )
    parser.add_argument(
        "--early_stopping_patience",
        "-e",
        required=False,
        type=int,
        default=5,
        help="Number of iterations to wait before early stopping",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-C",
        required=False,
        type=str,
        default="./checkpoints",
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--checkpoint_filename",
        "-c",
        required=False,
        type=str,
        default="kd_coco",
        help="Base filename for checkpoint files",
    )
    parser.add_argument(
        "--base_checkpoint",
        "-bc",
        required=False,
        type=str,
        default="",
        help="Path to pre-trained Mobile CLIP checkpoint to initialize student model",
    )
    parser.add_argument(
        "--data_size",
        "-D",
        required=False,
        type=float,
        default=1.0,
        help="Fraction of data to train on (0.0-1.0)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        "-a",
        required=False,
        type=int,
        default=2,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--use_swa",
        "-s",
        required=False,
        type=int,
        default=0,
        help="Whether to use Stochastic Weight Averaging (0/1)",
    )
    parser.add_argument(
        "--load_pretrained_checkpoint",
        "-l",
        required=False,
        type=str,
        default="",
        help="Path to previous checkpoint to resume training",
    )

    # Knowledge distillation specific parameters
    parser.add_argument(
        "--teacher_model",
        "-T",
        required=False,
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Teacher model name or path (HuggingFace model ID)",
    )
    parser.add_argument(
        "--original_weight",
        "-ow",
        required=False,
        type=float,
        default=0.7,
        help="Weight for original CLIP loss",
    )
    parser.add_argument(
        "--img_distill_weight",
        "-iw",
        required=False,
        type=float,
        default=0.1,
        help="Weight for image feature distillation loss",
    )
    parser.add_argument(
        "--txt_distill_weight",
        "-tw",
        required=False,
        type=float,
        default=0.1,
        help="Weight for text feature distillation loss",
    )
    parser.add_argument(
        "--response_weight",
        "-rw",
        required=False,
        type=float,
        default=0.1,
        help="Weight for response distillation loss",
    )
    parser.add_argument(
        "--distill_temperature",
        "-t",
        required=False,
        type=float,
        default=4.0,
        help="Temperature parameter for distillation",
    )

    # Debugging flag
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode (prints extra information)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    config_path = args.config_path
    max_epochs = args.max_epochs
    early_stopping_patience = args.early_stopping_patience
    checkpoint_dir = args.checkpoint_dir
    checkpoint_filename = args.checkpoint_filename
    base_checkpoint = args.base_checkpoint
    data_size = args.data_size
    accumulate_grad_batches = args.accumulate_grad_batches
    use_swa = args.use_swa
    load_pretrained_checkpoint = args.load_pretrained_checkpoint
    debug_mode = args.debug

    # Validate arguments
    assert 0 < data_size <= 1, "Expected data_size to be within range (0, 1]"
    assert use_swa in [0, 1], "Expected use_swa to be either 0 or 1"

    # Load and process data
    print("Loading data from CSV files...")
    train_csv_data = pol.read_csv(train_data_path)
    val_csv_data = pol.read_csv(val_data_path)

    train_csv_data = train_csv_data.select(
        ["file_name", "image_path", "caption"]
    ).sample(fraction=data_size, seed=42)
    val_csv_data = val_csv_data.select(["file_name", "image_path", "caption"]).sample(
        fraction=data_size, seed=42
    )
    train_csv_data = train_csv_data.to_pandas()
    val_csv_data = val_csv_data.to_pandas()

    # Load configuration


    print(f"Loading config from: {config_path}")
    config = {}
    with open(config_path, "r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(f"Error in config file: {exc}")
            sys.exit(1)


    # Load automatic KD configuration
    auto_kd_config_path = args.auto_kd_config
    auto_kd_config = load_automatic_kd_config(auto_kd_config_path)

    # Merge automatic KD config with main config
    if auto_kd_config:
        print(f"🤖 Loading automatic KD config from: {auto_kd_config_path}")
        config.update(auto_kd_config)
        # For auto KD, ensure knowledge_distillation section exists for base_checkpoint
        if "knowledge_distillation" not in config:
            config["knowledge_distillation"] = {}
    else:
        print("⚠️ No automatic KD config found, using manual parameters")
        # Set up manual KD parameters from command line arguments
        if "knowledge_distillation" not in config:
            config["knowledge_distillation"] = {}

        kd_config = config["knowledge_distillation"]

        # Use config values if they exist, otherwise use command line args
        config["knowledge_distillation"]["teacher_model"] = kd_config.get("teacher_model", args.teacher_model)
        config["knowledge_distillation"]["original_weight"] = kd_config.get("original_weight", args.original_weight)
        config["knowledge_distillation"]["img_distill_weight"] = kd_config.get("img_distill_weight",
                                                                               args.img_distill_weight)
        config["knowledge_distillation"]["txt_distill_weight"] = kd_config.get("txt_distill_weight",
                                                                               args.txt_distill_weight)
        config["knowledge_distillation"]["response_weight"] = kd_config.get("response_weight", args.response_weight)
        config["knowledge_distillation"]["distill_temperature"] = kd_config.get("distill_temperature",
                                                                                args.distill_temperature)

    # Add base checkpoint to both auto and manual configurations
    if base_checkpoint:
        config["knowledge_distillation"]["base_checkpoint"] = base_checkpoint

    # Create tokenizer
    print("Creating tokenizer...")
    if config["text_model_name"] != "convbert":
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer.padding_side = "left"

        # Ensure text model max_seq_length matches teacher model (77 for CLIP)
        if config["text_model"]["max_seq_length"] != 77:
            print(f"⚠️ Warning: Changing max_seq_length from {config['text_model']['max_seq_length']} to 77 to match CLIP")
            config["text_model"]["max_seq_length"] = 77
    else:
        tokenizer = ConvBertTokenizer.from_pretrained("YituTech/conv-bert-base")

    # Create data transformations
    print("Setting up data transformations...")
    train_transforms = alb.Compose(
        [
            alb.SmallestMaxSize(256, always_apply=True),
            CenterSquareCrop(224),
            alb.ColorJitter(),
            HorizontalFlip(),
            alb.Resize(224, 224, always_apply=True),
            alb.Normalize(
                mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, always_apply=True
            ),
        ]
    )

    val_transforms = alb.Compose(
        [
            alb.Resize(224, 224, always_apply=True),
            alb.Normalize(
                mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, always_apply=True
            ),
        ]
    )

    # Create datasets
    print("Creating datasets...")
    train_ds = CocoDataset(
        data=train_csv_data,
        text_tokenizer=tokenizer,
        transformations=train_transforms,
        config=config,
    )

    val_ds = CocoDataset(
        data=val_csv_data,
        text_tokenizer=tokenizer,
        transformations=val_transforms,
        config=config,
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_dl = td.DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dl = td.DataLoader(
        val_ds,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Debug first batch if enabled
    if debug_mode:
        print("Checking first batch...")
        for batch in train_dl:
            debug_first_batch(batch, config)
            break

    # Create model
    print("Creating KD model...")
    print(f"🔍 Passing base_checkpoint to model: '{base_checkpoint}'")
    #model = LitMobileCLiPKD(config, base_checkpoint=base_checkpoint)
    if auto_kd_config:
        print("🚀 Using Automatic Knowledge Distillation")
        model = LitMobileCLiPAutoKD(config, base_checkpoint=base_checkpoint)
    else:
        print("🔧 Using Manual Knowledge Distillation")
        model = LitMobileCLiPKD(config, base_checkpoint=base_checkpoint)

    # Setup callbacks
    print("Setting up training callbacks...")
    callbacks = []

    # Checkpointing
    model_chkpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename=f"{checkpoint_filename}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        verbose=True,
    )
    callbacks.append(model_chkpt)

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        verbose=True
    )
    callbacks.append(early_stop)

    # Rich progress bar
    rich_prog_bar = RichProgressBar()
    callbacks.append(rich_prog_bar)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # SWA if enabled
    if use_swa:
        print("Enabling Stochastic Weight Averaging...")
        swa = StochasticWeightAveraging(
            swa_epoch_start=0.7,  # Start at 70% of training
            swa_lrs=config.get("lr", 1e-4) / 10,
            annealing_epochs=5
        )
        callbacks.append(swa)

    # Setup logger
    print("Setting up WandB logger...")
    logger = WandbLogger(
        project="MobileCLIP-KD",
        name=f"kd-{config['image_model']['model_name']}-{args.teacher_model.split('/')[-1]}",
        log_model=True,
    )
    logger.watch(model, log="all", log_freq=100)

    # Create trainer
    print("Creating trainer...")
    if auto_kd_config and "max_steps" in config:
        print(f"🎯 Training with automatic KD for {config['max_steps']} steps")
        trainer = pl.Trainer(
            accelerator="cuda",
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            devices=torch.cuda.device_count(),
            precision=32,  # "16-mixed",
            max_steps=config["max_steps"],  # Use steps instead of epochs
            callbacks=callbacks,  # Use ALL callbacks, not just a subset
            logger=logger,
            gradient_clip_val=config.get("clip_grad_val", 1.0),
            gradient_clip_algorithm="value",
            val_check_interval=500,  # Validate every 500 steps
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=10,
        )
    else:
        print(f"🎯 Training with manual KD for {max_epochs} epochs")
        trainer = pl.Trainer(
            accelerator="cuda",
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            devices=torch.cuda.device_count(),
            precision=32,  # "16-mixed",
            max_epochs=max_epochs,
            callbacks=callbacks,  # Use ALL callbacks
            logger=logger,
            gradient_clip_val=config.get("clip_grad_val", 1.0),
            gradient_clip_algorithm="value",
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=10,
        )
    # trainer = pl.Trainer(
    #     accelerator="cuda",
    #     strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
    #     devices=torch.cuda.device_count(),
    #     precision=32, ##"16-mixed"
    #     max_epochs=max_epochs,
    #     callbacks=callbacks,
    #     logger=logger,
    #     gradient_clip_val=config.get("clip_grad_val", 1.0),
    #     gradient_clip_algorithm="value",
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     log_every_n_steps=10,
    # )

    # Train the model
    print("Starting training...")
    ckpt_path = None if load_pretrained_checkpoint == "" else load_pretrained_checkpoint
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=ckpt_path,
    )

    # Get the best model path
    best_model_path = model_chkpt.best_model_path
    print(f"\nTraining complete! Best model saved to: {best_model_path}")


