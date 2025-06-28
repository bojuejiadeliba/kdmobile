"""
PyTorch Lightning module for automatic knowledge distillation
Integrates with auto_kd_training.py for progressive training
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from models.mobile_clip_kd import MobileCLIPWithKD
from auto_kd_training import SmartKDTrainingLoop, setup_automatic_kd_training


class LitMobileCLiPAutoKD(pl.LightningModule):
    """
    Lightning module that uses automatic knowledge distillation with progressive training
    """

    def __init__(self, config, base_checkpoint=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['base_checkpoint'])

        # Load automatic KD config if provided
        if "automatic_kd" in config:
            self.auto_kd_config = config["automatic_kd"]
        else:
            # Use defaults from your config file
            self.auto_kd_config = {
                "phase_1_steps": 1000,
                "phase_2_steps": 2000,
                "phase_3_steps": 3000,
                "adaptation_rate": 0.01,
                "min_temperature": 4.0,
                "max_temperature": 8.0,
                "plateau_patience": 50,
                "gradient_check_frequency": 10,
                "use_cosine_similarity": True,
                "normalize_losses": True,
                "smooth_weight_updates": True
            }

        print("🤖 Initializing Automatic Knowledge Distillation")
        print(f"📋 Auto KD Config: {self.auto_kd_config}")

        # Pass base checkpoint to config if provided
        if base_checkpoint:
            print(f"🔄 Loading base checkpoint: {base_checkpoint}")
            if "knowledge_distillation" not in config:
                config["knowledge_distillation"] = {}
            config["knowledge_distillation"]["base_checkpoint"] = base_checkpoint

        # Initialize the KD model
        teacher_model = config.get("knowledge_distillation", {}).get("teacher_model", "openai/clip-vit-base-patch32")
        self.clip_model = MobileCLIPWithKD(config, teacher_model=teacher_model)

        # Training step counter for progressive training
        self.training_step_count = 0

        # Initialize smart training components (will be set up in configure_optimizers)
        self.smart_training_loop = None

    def forward(self, image, text, attn_mask=None, neg_text=None, neg_attn_mask=None, neg_image=None):
        return self.clip_model(image, text, attn_mask, neg_text, neg_attn_mask, neg_image)

    def _validate_inputs(self, batch):
        """Basic input validation"""
        img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]

        if torch.isnan(img).any() or torch.isnan(txt["input_ids"]).any() or torch.isnan(neg_txt["input_ids"]).any():
            return False
        return True

    def training_step(self, batch, batch_idx):
        """Training step using simplified automatic KD logic"""

        # Input validation
        if not self._validate_inputs(batch):
            print(f"⚠️ Invalid inputs at step {self.training_step_count}")
            return None

        try:
            # Extract data
            img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]
            txt_input_ids = txt["input_ids"].squeeze()
            txt_attn_mask = txt["attention_mask"].squeeze().float()
            neg_txt_input_ids = neg_txt["input_ids"].squeeze()
            neg_txt_attn_mask = neg_txt["attention_mask"].squeeze().float()

            # Forward pass through KD model
            outputs = self.clip_model(
                img, txt_input_ids, txt_attn_mask,
                neg_txt_input_ids, neg_txt_attn_mask
            )

            # Progressive training logic based on current step
            current_phase = self._get_current_phase()

            # Get losses from outputs
            original_loss = outputs.get("original_loss", outputs.get("Em", torch.tensor(0.0)) - outputs.get("Ej", torch.tensor(0.0)))

            # Progressive loss combination with safe attribute access
            if self.training_step_count < self.auto_kd_config["phase_1_steps"]:
                # Phase 1: Original loss only
                total_loss = original_loss
                self.log("train_phase", 1.0, on_step=False, on_epoch=True)

            elif self.training_step_count < self.auto_kd_config["phase_2_steps"]:
                # Phase 2: Original + feature distillation
                img_distill_loss = outputs.get("img_distill_loss", torch.tensor(0.0, device=self.device))
                txt_distill_loss = outputs.get("txt_distill_loss", torch.tensor(0.0, device=self.device))

                # Use safe attribute access
                orig_weight = getattr(self, 'original_weight', 0.7)
                img_weight = getattr(self, 'img_distill_weight', 0.15)
                txt_weight = getattr(self, 'txt_distill_weight', 0.15)

                total_loss = (orig_weight * original_loss +
                             img_weight * img_distill_loss +
                             txt_weight * txt_distill_loss)

                self.log("train_phase", 2.0, on_step=False, on_epoch=True)
                self.log("train_img_distill_loss", img_distill_loss, on_step=False, on_epoch=True)
                self.log("train_txt_distill_loss", txt_distill_loss, on_step=False, on_epoch=True)

            else:
                # Phase 3: Full distillation
                img_distill_loss = outputs.get("img_distill_loss", torch.tensor(0.0, device=self.device))
                txt_distill_loss = outputs.get("txt_distill_loss", torch.tensor(0.0, device=self.device))
                response_loss = outputs.get("response_loss", torch.tensor(0.0, device=self.device))

                # Use safe attribute access
                orig_weight = getattr(self, 'original_weight', 0.6)
                img_weight = getattr(self, 'img_distill_weight', 0.15)
                txt_weight = getattr(self, 'txt_distill_weight', 0.15)
                resp_weight = getattr(self, 'response_weight', 0.1)

                total_loss = (orig_weight * original_loss +
                             img_weight * img_distill_loss +
                             txt_weight * txt_distill_loss +
                             resp_weight * response_loss)

                self.log("train_phase", 3.0, on_step=False, on_epoch=True)
                self.log("train_img_distill_loss", img_distill_loss, on_step=False, on_epoch=True)
                self.log("train_txt_distill_loss", txt_distill_loss, on_step=False, on_epoch=True)
                self.log("train_response_loss", response_loss, on_step=False, on_epoch=True)

            # Log metrics
            self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_original_loss", original_loss, on_step=False, on_epoch=True)
            self.log("train_step_count", float(self.training_step_count), on_step=False, on_epoch=True)

            self.training_step_count += 1
            return total_loss

        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return self._manual_training_step(batch, batch_idx)

    def _manual_training_step(self, batch, batch_idx):
        """Fallback manual training step"""
        try:
            # Extract data
            img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]
            txt_input_ids = txt["input_ids"].squeeze()
            txt_attn_mask = txt["attention_mask"].squeeze().float()
            neg_txt_input_ids = neg_txt["input_ids"].squeeze()
            neg_txt_attn_mask = neg_txt["attention_mask"].squeeze().float()

            # Forward pass
            outputs = self.clip_model(
                img, txt_input_ids, txt_attn_mask,
                neg_txt_input_ids, neg_txt_attn_mask
            )

            # Simple loss calculation
            original_loss = outputs.get("original_loss", outputs.get("Em", torch.tensor(0.0)) - outputs.get("Ej", torch.tensor(0.0)))

            # Log basic metrics
            self.log("train_loss", original_loss, prog_bar=True, on_step=True, on_epoch=True)
            self.training_step_count += 1

            return original_loss

        except Exception as e:
            print(f"❌ Manual training step failed: {e}")
            return None

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        if not self._validate_inputs(batch):
            return None

        try:
            # Extract data
            img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]
            txt_input_ids = txt["input_ids"].squeeze()
            txt_attn_mask = txt["attention_mask"].squeeze().float()

            # Forward pass (no negative examples needed for validation)
            outputs = self.clip_model(img, txt_input_ids, txt_attn_mask)

            # Validation loss
            val_loss = outputs.get("original_loss", outputs.get("Em", torch.tensor(0.0)) - outputs.get("Ej", torch.tensor(0.0)))

            self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

            return val_loss

        except Exception as e:
            print(f"❌ Validation step failed: {e}")
            return None

    def configure_optimizers(self):
        """Configure optimizers - simplified to avoid conflicts"""
        try:
            # Get basic optimizer settings
            base_lr = self.config.get("base_lr", 1e-4)
            weight_decay = self.config.get("weight_decay", 1e-5)

            # Ensure they are floats
            if isinstance(base_lr, str):
                base_lr = float(base_lr)
            if isinstance(weight_decay, str):
                weight_decay = float(weight_decay)

            # Create optimizer with different learning rates for different components
            optimizer = torch.optim.AdamW([
                {
                    "params": self.clip_model.student.text_model.parameters(),
                    "lr": base_lr * 0.5,
                    "weight_decay": weight_decay * 0.1,
                },
                {
                    "params": self.clip_model.student.img_model.parameters(),
                    "lr": base_lr * 0.7,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(self.clip_model.student.img_projection.parameters()) +
                              list(self.clip_model.student.text_projection.parameters()),
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(self.clip_model.img_align.parameters()) +
                              list(self.clip_model.txt_align.parameters()),
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ])

            # Simple scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.auto_kd_config.get("phase_3_steps", 3000),
                eta_min=1e-7
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }

        except Exception as e:
            print(f"⚠️ Optimizer setup failed: {e}, using fallback")
            return self._configure_fallback_optimizer()

    def _configure_fallback_optimizer(self):
        """Fallback optimizer configuration"""
        base_lr = self.config.get("base_lr", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)

        # Ensure learning rate is a float
        if isinstance(base_lr, str):
            base_lr = float(base_lr)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.auto_kd_config.get("phase_3_steps", 3000),
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def on_validation_epoch_end(self):
        """Update automatic KD with validation performance"""
        if self.smart_training_loop is not None:
            try:
                # Get average validation loss
                val_loss = self.trainer.callback_metrics.get("val_loss", float('inf'))
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()

                # Update smart training loop with validation performance
                self.smart_training_loop.kd_trainer.loss_balancer.update_weights(
                    losses={},
                    val_performance=val_loss,
                    step=self.training_step_count
                )

            except Exception as e:
                print(f"⚠️ Failed to update KD with validation performance: {e}")

    def on_train_epoch_end(self):
        """Log training progress"""
        current_phase = self._get_current_phase()
        print(f"📊 Epoch complete - Training Step: {self.training_step_count}, Phase: {current_phase}")

    def _get_current_phase(self):
        """Get current training phase"""
        if self.training_step_count < self.auto_kd_config["phase_1_steps"]:
            return "1 (Original Only)"
        elif self.training_step_count < self.auto_kd_config["phase_2_steps"]:
            return "2 (+ Feature Distillation)"
        else:
            return "3 (Full Distillation)"


def load_automatic_kd_config(config_path):
    """Load automatic KD configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Failed to load auto KD config: {e}")
        return {}