"""
PyTorch Lightning module for training MobileCLIP with knowledge distillation
Enhanced with NaN protection and gradient stability fixes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import pytorch_lightning as pl
from models.mobile_clip_kd import MobileCLIPWithKD


# ================================
# NaN PROTECTION UTILITIES
# ================================

def safe_gradient_clipping(model, max_norm=1.0):
    """Safe gradient clipping with NaN detection"""
    nan_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads.append(name)
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    if nan_grads and len(nan_grads) <= 3:
        print(f"⚠️ Found NaN gradients in: {nan_grads}")

    grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm, len(nan_grads) > 0


def reset_batchnorm_stats(model):
    """Reset problematic BatchNorm statistics"""
    reset_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                if (torch.abs(module.running_mean).max() > 100 or
                    torch.abs(module.running_var).max() > 1000):
                    module.reset_running_stats()
                    reset_count += 1

    if reset_count > 0:
        print(f"🔧 Reset {reset_count} BatchNorm layers")
    return reset_count



# ================================
# ENHANCED LIGHTNING MODULE
# ================================

class LitMobileCLiPKD(pl.LightningModule):
    def __init__(self, config, base_checkpoint=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['base_checkpoint'])

        # Ensure knowledge_distillation section exists
        if "knowledge_distillation" not in config:
            config["knowledge_distillation"] = {}

        # Pass base checkpoint to config if provided
        if base_checkpoint:
            print(f"🔄 Initializing KD model with base checkpoint: {base_checkpoint}")
            config["knowledge_distillation"]["base_checkpoint"] = base_checkpoint
        else:
            print("🆕 Initializing KD model from scratch")

        # Initialize the KD model
        teacher_model = config.get("knowledge_distillation", {}).get("teacher_model", "openai/clip-vit-base-patch32")
        self.clip_model = MobileCLIPWithKD(config, teacher_model=teacher_model)

        # Loss weights from config
        kd_config = config.get("knowledge_distillation", {})
        self.original_weight = kd_config.get("original_weight", 0.6)
        self.img_distill_weight = kd_config.get("img_distill_weight", 0.15)
        self.txt_distill_weight = kd_config.get("txt_distill_weight", 0.15)
        self.response_weight = kd_config.get("response_weight", 0.1)

        print(f"🎯 KD Loss weights: orig={self.original_weight}, img={self.img_distill_weight}, "
              f"txt={self.txt_distill_weight}, resp={self.response_weight}")

        # Training monitoring
        self.nan_batches = 0
        self.total_batches = 0

    def forward(self, image, text, attn_mask=None, neg_text=None, neg_attn_mask=None, neg_image=None):
        return self.clip_model(image, text, attn_mask, neg_text, neg_attn_mask, neg_image)

    def _validate_inputs(self, batch):
        """Basic input validation"""
        img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]

        if torch.isnan(img).any() or torch.isnan(txt["input_ids"]).any() or torch.isnan(neg_txt["input_ids"]).any():
            return False
        return True

    def _common_steps(self, batch, batch_idx):
        self.total_batches += 1

        # Periodic maintenance
        if batch_idx > 0 and batch_idx % 500 == 0:
            reset_count = reset_batchnorm_stats(self.clip_model)

        # Input validation
        if not self._validate_inputs(batch):
            self.nan_batches += 1
            return None

        # Extract data
        img, txt, neg_txt = batch["img"], batch["txt"], batch["neg_txt"]
        txt_input_ids = txt["input_ids"].squeeze()
        txt_attn_mask = txt["attention_mask"].squeeze().float()
        neg_txt_input_ids = neg_txt["input_ids"].squeeze()
        neg_txt_attn_mask = neg_txt["attention_mask"].squeeze().float()

        # Forward pass
        try:
            outputs = self.clip_model(
                img, txt_input_ids, txt_attn_mask,
                neg_image=img, neg_text=neg_txt_input_ids, neg_attn_mask=neg_txt_attn_mask
            )

            if torch.isnan(outputs["Ej"]).any() or torch.isnan(outputs["Em"]).any():
                self.nan_batches += 1
                return None

        except Exception as e:
            if batch_idx % 100 == 0:  # Only log occasionally
                print(f"❌ Model forward pass failed: {e}")
            self.nan_batches += 1
            return None

        # Calculate original loss
        original_loss = outputs["Em"] - outputs["Ej"]

        if torch.isnan(original_loss) or torch.isinf(original_loss):
            self.nan_batches += 1
            return None

        # Knowledge distillation losses
        if "teacher_image_embeds" in outputs and self.training:
            try:
                batch_size = img.size(0)
                student_img_proj = outputs["img_proj"][:batch_size]
                student_txt_proj = outputs["txt_proj"][:batch_size]

                # Feature distillation
                img_distill_loss, txt_distill_loss = self.clip_model.feature_distillation_loss(
                    student_img_proj, student_txt_proj,
                    outputs["teacher_image_embeds"], outputs["teacher_text_embeds"]
                )

                # Response distillation
                response_loss = self.clip_model.response_distillation_loss(
                    student_img_proj, student_txt_proj, outputs["teacher_logits_per_image"]
                )

                # Combined loss
                total_loss = (
                        self.original_weight * original_loss +
                        self.img_distill_weight * img_distill_loss +
                        self.txt_distill_weight * txt_distill_loss +
                        self.response_weight * response_loss
                )

                # Log individual losses (less frequent)
                if batch_idx % 10 == 0:
                    self.log("loss/original", original_loss, on_step=True)
                    self.log("loss/img_distill", img_distill_loss, on_step=True)
                    self.log("loss/txt_distill", txt_distill_loss, on_step=True)
                    self.log("loss/response", response_loss, on_step=True)

            except Exception as e:
                if batch_idx % 100 == 0:
                    print(f"⚠️ Distillation computation failed: {e}, using original loss")
                total_loss = original_loss
        else:
            total_loss = original_loss

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._common_steps(batch, batch_idx)

        if loss is None:
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_loss", dummy_loss, on_step=True)
            return dummy_loss

        self.log("train_loss", loss, on_step=True)

        # Log NaN ratio occasionally
        if batch_idx % 200 == 0 and self.total_batches > 0:
            nan_ratio = self.nan_batches / self.total_batches
            self.log("nan_batch_ratio", nan_ratio, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_steps(batch, batch_idx)

        if loss is None:
            dummy_loss = torch.tensor(1.0, device=self.device, requires_grad=False)
            self.log("val_loss", dummy_loss, prog_bar=True)
            return dummy_loss

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Gradient monitoring and clipping"""
        if self.training and outputs is not None:
            grad_norm, has_nan_grads = safe_gradient_clipping(self.clip_model, max_norm=1.0)

            if batch_idx % 50 == 0:  # Less frequent logging
                self.log("grad_norm", grad_norm, on_step=True)
                if has_nan_grads:
                    self.log("nan_gradients", 1.0, on_step=True)

    def configure_optimizers(self):
        main_lr = self.config.get("lr", 2e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)

        optimizer = torch.optim.AdamW([
            {
                "params": self.clip_model.student.text_model.parameters(),
                "lr": main_lr * 0.5,
                "weight_decay": weight_decay * 0.1,
            },
            {
                "params": self.clip_model.student.img_model.parameters(),
                "lr": main_lr * 0.7,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.clip_model.student.img_projection.parameters()) +
                          list(self.clip_model.student.text_projection.parameters()),
                "lr": main_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.clip_model.img_align.parameters()) +
                          list(self.clip_model.txt_align.parameters()),
                "lr": main_lr,
                "weight_decay": weight_decay,
            }
        ])

        if "T_0" in self.config:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.get("T_0", 50),
                eta_min=self.config.get("min_lr", 1e-6)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
            }

        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # Remove teacher model from checkpoint to save space
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if k.startswith("clip_model.teacher")]
        for key in keys_to_remove:
            del checkpoint["state_dict"][key]

        checkpoint["training_stats"] = {
            "nan_batches": self.nan_batches,
            "total_batches": self.total_batches
        }

    def on_load_checkpoint(self, checkpoint):
        stats = checkpoint.get("training_stats", {})
        self.nan_batches = stats.get("nan_batches", 0)
        self.total_batches = stats.get("total_batches", 0)

    def on_train_epoch_start(self):
        # ✅ FIXED: Add proper logging and make the reset conditional
        if self.current_epoch % 5 == 0:
            print(f"🔧 Performing periodic maintenance at epoch {self.current_epoch}")
            reset_count = reset_batchnorm_stats(self.clip_model)

    def on_train_epoch_end(self):
        # ✅ FIXED: Better counter management
        if self.total_batches > 0:
            nan_ratio = self.nan_batches / self.total_batches
            print(
                f"📊 Epoch {self.current_epoch}: {self.nan_batches}/{self.total_batches} NaN batches ({nan_ratio:.2%})")

            if nan_ratio > 0.1:
                print(f"⚠️ High NaN ratio detected!")

        # Reset counters for next epoch
        self.nan_batches = 0
        self.total_batches = 0