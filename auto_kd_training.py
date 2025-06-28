"""
Automatic Knowledge Distillation Implementation with Adaptive Loss Weighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import math


class AutomaticLossBalancer(nn.Module):
    """
    Automatically balances multiple loss components based on:
    1. Loss magnitude differences
    2. Gradient norm ratios
    3. Training progress
    4. Validation performance
    """

    def __init__(self, loss_names, initial_weights=None, adaptation_rate=0.01):
        super().__init__()
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.adaptation_rate = adaptation_rate

        # Initialize weights
        if initial_weights is None:
            initial_weights = [1.0 / self.num_losses] * self.num_losses

        # Use learnable parameters for weights (in log space for stability)
        self.log_weights = nn.Parameter(torch.tensor(initial_weights).log())

        # Track loss history for adaptation
        self.loss_history = {name: deque(maxlen=100) for name in loss_names}
        self.grad_norm_history = {name: deque(maxlen=100) for name in loss_names}

        # Performance tracking
        self.best_val_performance = float('inf')
        self.performance_plateau_count = 0

        # Adaptation strategy
        self.strategy = "gradient_norm"  # Options: "magnitude", "gradient_norm", "uncertainty"

    @property
    def weights(self):
        """Get normalized weights from log parameters"""
        return F.softmax(self.log_weights, dim=0)

    def compute_adaptive_weights(self, losses, gradients=None, step=None):
        """
        Compute adaptive weights based on current losses and gradients

        Args:
            losses: Dict of {loss_name: loss_value}
            gradients: Dict of {loss_name: gradient_norm} (optional)
            step: Current training step
        """
        if self.strategy == "magnitude":
            return self._magnitude_based_weighting(losses)
        elif self.strategy == "gradient_norm":
            return self._gradient_norm_weighting(losses, gradients)
        elif self.strategy == "uncertainty":
            return self._uncertainty_weighting(losses, step)
        else:
            return self.weights

    def _magnitude_based_weighting(self, losses):
        """Weight inversely proportional to loss magnitude"""
        loss_values = torch.tensor([losses[name].item() for name in self.loss_names])

        # Avoid division by zero
        loss_values = torch.clamp(loss_values, min=1e-8)

        # Inverse weighting (smaller losses get higher weights)
        inverse_weights = 1.0 / loss_values
        normalized_weights = inverse_weights / inverse_weights.sum()

        return normalized_weights

    def _gradient_norm_weighting(self, losses, gradients):
        """Weight based on gradient norms to balance learning rates"""
        if gradients is None:
            return self.weights

        grad_norms = torch.tensor([gradients.get(name, 1.0) for name in self.loss_names])
        grad_norms = torch.clamp(grad_norms, min=1e-8)

        # Target: equal gradient norms for all losses
        target_norm = grad_norms.mean()
        weight_adjustments = target_norm / grad_norms

        # Smooth adjustment
        current_weights = self.weights
        new_weights = current_weights * weight_adjustments
        normalized_weights = new_weights / new_weights.sum()

        return normalized_weights

    def _uncertainty_weighting(self, losses, step):
        """Weight based on loss uncertainty and learning progress"""
        weights = self.weights.clone()

        # Compute loss variance for each component
        for i, name in enumerate(self.loss_names):
            loss_val = losses[name].item()
            self.loss_history[name].append(loss_val)

            if len(self.loss_history[name]) > 10:
                recent_losses = list(self.loss_history[name])[-10:]
                variance = np.var(recent_losses)

                # Higher variance = more uncertain = lower weight
                uncertainty_factor = 1.0 / (1.0 + variance)
                weights[i] *= uncertainty_factor

        return weights / weights.sum()

    def update_weights(self, losses, gradients=None, val_performance=None, step=None):
        """Update weights based on training progress"""

        # Compute adaptive weights
        adaptive_weights = self.compute_adaptive_weights(losses, gradients, step)

        # Smooth update to avoid oscillations
        current_weights = self.weights
        new_weights = (1 - self.adaptation_rate) * current_weights + \
                      self.adaptation_rate * adaptive_weights

        # Update parameters
        self.log_weights.data = new_weights.log()

        # Track validation performance for meta-adaptation
        if val_performance is not None:
            if val_performance < self.best_val_performance:
                self.best_val_performance = val_performance
                self.performance_plateau_count = 0
            else:
                self.performance_plateau_count += 1

                # If plateauing, try different adaptation strategy
                if self.performance_plateau_count > 50:
                    self._switch_adaptation_strategy()

    def _switch_adaptation_strategy(self):
        """Switch adaptation strategy if performance plateaus"""
        strategies = ["magnitude", "gradient_norm", "uncertainty"]
        current_idx = strategies.index(self.strategy)
        self.strategy = strategies[(current_idx + 1) % len(strategies)]
        print(f"🔄 Switching to {self.strategy} weighting strategy")
        self.performance_plateau_count = 0


class ProgressiveKDTrainer(nn.Module):
    """
    Progressive Knowledge Distillation Trainer with Automatic Loss Balancing
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Training phases
        self.phase_1_steps = config.get("phase_1_steps", 1000)  # Original loss only
        self.phase_2_steps = config.get("phase_2_steps", 2000)  # + Feature distillation
        self.phase_3_steps = config.get("phase_3_steps", 3000)  # + Response distillation

        # Automatic loss balancer
        loss_names = ["original", "img_feature", "txt_feature", "response"]
        initial_weights = [0.8, 0.1, 0.1, 0.0]  # Start conservative
        self.loss_balancer = AutomaticLossBalancer(
            loss_names=loss_names,
            initial_weights=initial_weights,
            adaptation_rate=config.get("adaptation_rate", 0.01)
        )

        # Temperature scheduling
        self.register_buffer('temperature', torch.tensor(8.0))
        self.min_temperature = config.get("min_temperature", 4.0)
        self.max_temperature = config.get("max_temperature", 8.0)

        # Performance tracking
        self.step_count = 0
        self.loss_history = []

    def forward(self, student_outputs, step, gradients=None, val_performance=None):
        """
        Progressive training with automatic loss balancing

        Args:
            student_outputs: Dictionary containing model outputs
            step: Current training step
            gradients: Optional gradient norms for each loss component
            val_performance: Optional validation performance metric
        """
        self.step_count = step

        # Update temperature based on step
        self._update_temperature(step)

        # Compute individual losses
        losses = self._compute_individual_losses(student_outputs, step)

        # Determine active losses based on training phase
        active_losses = self._get_active_losses(step)

        # Update loss weights
        self.loss_balancer.update_weights(
            losses=losses,
            gradients=gradients,
            val_performance=val_performance,
            step=step
        )

        # Combine losses with automatic weights
        total_loss = self._combine_losses(losses, active_losses)

        return total_loss, losses, self.loss_balancer.weights

    def _update_temperature(self, step):
        """Update temperature based on training progress"""
        if step < self.phase_1_steps:
            self.temperature = torch.tensor(self.max_temperature)
        elif step < self.phase_2_steps:
            # Linear decay from max to min
            progress = (step - self.phase_1_steps) / (self.phase_2_steps - self.phase_1_steps)
            temp = self.max_temperature - progress * (self.max_temperature - self.min_temperature)
            self.temperature = torch.tensor(temp)
        else:
            self.temperature = torch.tensor(self.min_temperature)

    def _compute_individual_losses(self, student_outputs, step):
        """Compute all loss components individually"""
        losses = {}

        # Original contrastive loss
        losses["original"] = student_outputs["Em"] - student_outputs["Ej"]

        # Feature distillation losses
        if "aligned_img_feats" in student_outputs and step >= self.phase_1_steps:
            img_loss, txt_loss = self._compute_feature_distillation(student_outputs)
            losses["img_feature"] = img_loss
            losses["txt_feature"] = txt_loss
        else:
            losses["img_feature"] = torch.tensor(0.0, device=losses["original"].device)
            losses["txt_feature"] = torch.tensor(0.0, device=losses["original"].device)

        # Response distillation loss
        if step >= self.phase_2_steps:
            resp_loss = self._compute_response_distillation(student_outputs)
            losses["response"] = resp_loss
        else:
            losses["response"] = torch.tensor(0.0, device=losses["original"].device)

        return losses

    def _compute_feature_distillation(self, student_outputs):
        """Compute feature-level distillation losses"""
        aligned_img = student_outputs["aligned_img_feats"]
        aligned_txt = student_outputs["aligned_txt_feats"]
        teacher_img = student_outputs["teacher_image_embeds"]
        teacher_txt = student_outputs["teacher_text_embeds"]

        # Ensure same batch size
        min_batch = min(aligned_img.size(0), teacher_img.size(0))
        aligned_img = F.normalize(aligned_img[:min_batch], p=2, dim=-1)
        aligned_txt = F.normalize(aligned_txt[:min_batch], p=2, dim=-1)
        teacher_img = F.normalize(teacher_img[:min_batch], p=2, dim=-1)
        teacher_txt = F.normalize(teacher_txt[:min_batch], p=2, dim=-1)

        # Use cosine similarity loss (more stable than MSE)
        img_loss = 1 - F.cosine_similarity(aligned_img, teacher_img, dim=-1).mean()
        txt_loss = 1 - F.cosine_similarity(aligned_txt, teacher_txt, dim=-1).mean()

        return img_loss, txt_loss

    def _compute_response_distillation(self, student_outputs):
        """Compute response-level distillation loss"""
        try:
            aligned_img = student_outputs["aligned_img_feats"]
            aligned_txt = student_outputs["aligned_txt_feats"]
            teacher_img = student_outputs["teacher_image_embeds"]
            teacher_txt = student_outputs["teacher_text_embeds"]

            min_batch = min(aligned_img.size(0), teacher_img.size(0))
            aligned_img = F.normalize(aligned_img[:min_batch], p=2, dim=-1)
            aligned_txt = F.normalize(aligned_txt[:min_batch], p=2, dim=-1)
            teacher_img = F.normalize(teacher_img[:min_batch], p=2, dim=-1)
            teacher_txt = F.normalize(teacher_txt[:min_batch], p=2, dim=-1)

            # Compute similarity matrices
            student_sim = torch.matmul(aligned_img, aligned_txt.t()) * 100.0
            teacher_sim = torch.matmul(teacher_img, teacher_txt.t()) * 100.0

            # KL divergence with temperature
            kl_loss = F.kl_div(
                F.log_softmax(student_sim / self.temperature, dim=-1),
                F.softmax(teacher_sim / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)

            return kl_loss

        except Exception as e:
            print(f"Response distillation failed: {e}")
            return torch.tensor(0.0, device=aligned_img.device, requires_grad=True)

    def _get_active_losses(self, step):
        """Determine which losses are active at current step"""
        active = {"original": True}

        if step >= self.phase_1_steps:
            active["img_feature"] = True
            active["txt_feature"] = True
        else:
            active["img_feature"] = False
            active["txt_feature"] = False

        if step >= self.phase_2_steps:
            active["response"] = True
        else:
            active["response"] = False

        return active

    def _combine_losses(self, losses, active_losses):
        """Combine losses with automatic weights"""
        weights = self.loss_balancer.weights
        total_loss = torch.tensor(0.0, device=losses["original"].device)

        for i, (name, loss) in enumerate(losses.items()):
            if active_losses.get(name, False):
                total_loss += weights[i] * loss

        return total_loss


class SmartKDTrainingLoop:
    """
    Smart training loop with automatic loss balancing and monitoring
    """

    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Progressive KD trainer
        self.kd_trainer = ProgressiveKDTrainer(config)

        # Gradient tracking for automatic balancing
        self.gradient_tracker = {}

        # Performance tracking
        self.train_metrics = []
        self.val_metrics = []

    def train_step(self, batch, step):
        """Single training step with automatic KD"""
        self.model.train()

        try:
            # Forward pass
            outputs = self.model(
                batch["img"],
                batch["txt"]["input_ids"].squeeze(),
                batch["txt"]["attention_mask"].squeeze().float(),
                neg_image=batch["img"],
                neg_text=batch["neg_txt"]["input_ids"].squeeze(),
                neg_attn_mask=batch["neg_txt"]["attention_mask"].squeeze().float()
            )

            # Compute gradients for each loss component (for automatic balancing)
            gradients = self._compute_component_gradients(outputs, step)

            # Progressive KD with automatic balancing
            total_loss, individual_losses, weights = self.kd_trainer(
                outputs, step, gradients
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Check for NaN
            if torch.isnan(total_loss) or torch.isnan(grad_norm):
                print(f"⚠️ NaN detected at step {step}, skipping")
                return None

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Log progress
            if step % 50 == 0:
                self._log_progress(step, individual_losses, weights, grad_norm)

            return {
                'total_loss': total_loss.item(),
                'individual_losses': {k: v.item() for k, v in individual_losses.items()},
                'weights': weights.detach().cpu().numpy(),
                'grad_norm': grad_norm.item()
            }

        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return None

    def _compute_component_gradients(self, outputs, step):
        """Compute gradient norms for each loss component"""
        if step % 10 != 0:  # Only compute occasionally for efficiency
            return self.gradient_tracker

        gradients = {}

        # Original loss gradient
        original_loss = outputs["Em"] - outputs["Ej"]
        original_grad = torch.autograd.grad(
            original_loss, self.model.parameters(),
            retain_graph=True, create_graph=False, allow_unused=True
        )
        gradients["original"] = sum(g.norm().item() for g in original_grad if g is not None)

        # Feature loss gradients (if available)
        if "aligned_img_feats" in outputs and step >= 1000:
            try:
                img_loss, txt_loss = self.kd_trainer._compute_feature_distillation(outputs)

                img_grad = torch.autograd.grad(
                    img_loss, self.model.parameters(),
                    retain_graph=True, create_graph=False, allow_unused=True
                )
                gradients["img_feature"] = sum(g.norm().item() for g in img_grad if g is not None)

                txt_grad = torch.autograd.grad(
                    txt_loss, self.model.parameters(),
                    retain_graph=True, create_graph=False, allow_unused=True
                )
                gradients["txt_feature"] = sum(g.norm().item() for g in txt_grad if g is not None)

            except:
                gradients["img_feature"] = 1.0
                gradients["txt_feature"] = 1.0

        self.gradient_tracker = gradients
        return gradients

    def _log_progress(self, step, losses, weights, grad_norm):
        """Log training progress with automatic weights"""
        print(f"\n=== Step {step} ===")
        print(f"Losses:")
        for name, loss in losses.items():
            print(f"  {name}: {loss.item():.4f}")

        print(f"Automatic Weights:")
        for i, (name, weight) in enumerate(zip(["original", "img_feature", "txt_feature", "response"], weights)):
            print(f"  {name}: {weight:.3f}")

        print(f"Grad Norm: {grad_norm:.4f}")
        print(f"Temperature: {self.kd_trainer.temperature.item():.2f}")
        print(f"Phase: {self._get_current_phase(step)}")

    def _get_current_phase(self, step):
        """Get current training phase"""
        if step < self.kd_trainer.phase_1_steps:
            return "1 (Original Only)"
        elif step < self.kd_trainer.phase_2_steps:
            return "2 (+ Feature Distillation)"
        else:
            return "3 (Full Distillation)"

    def validate(self, val_dataloader, step):
        """Validation with performance tracking for automatic adaptation"""
        self.model.eval()
        total_val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                outputs = self.model(
                    batch["img"],
                    batch["txt"]["input_ids"].squeeze(),
                    batch["txt"]["attention_mask"].squeeze().float()
                )

                val_loss = outputs["Em"] - outputs["Ej"]
                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    num_batches += 1

        if num_batches > 0:
            avg_val_loss = total_val_loss / num_batches
            print(f"📊 Validation Loss: {avg_val_loss:.4f}")

            # Update KD trainer with validation performance
            self.kd_trainer.loss_balancer.update_weights(
                losses={}, val_performance=avg_val_loss, step=step
            )

            return avg_val_loss

        return float('inf')


# Usage Example
def setup_automatic_kd_training(config, base_checkpoint=None):
    """Setup automatic KD training"""

    # Update config with automatic KD settings
    config.update({
        "phase_1_steps": 1000,  # Original loss only
        "phase_2_steps": 2000,  # + Feature distillation
        "phase_3_steps": 3000,  # + Response distillation
        "adaptation_rate": 0.01,  # How fast to adapt weights
        "min_temperature": 4.0,  # Final temperature
        "max_temperature": 8.0,  # Initial temperature
    })

    # Create model
    from models.mobile_clip_kd import MobileCLIPWithKD
    model = MobileCLIPWithKD(config, base_checkpoint=base_checkpoint)

    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.clip_model.student.text_model.parameters(), 'lr': 1e-5},
        {'params': model.clip_model.student.img_model.parameters(), 'lr': 5e-6},
        {'params': list(model.clip_model.student.img_projection.parameters()) +
                   list(model.clip_model.student.text_projection.parameters()), 'lr': 1e-4},
        {'params': model.clip_model.img_align.parameters(), 'lr': 1e-4},
        {'params': model.clip_model.txt_align.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, eta_min=1e-7
    )

    # Smart training loop
    training_loop = SmartKDTrainingLoop(model, optimizer, scheduler, config)

    return training_loop


# Main training function
def train_with_automatic_kd(training_loop, train_dl, val_dl, max_steps=5000):
    """Main training with automatic knowledge distillation"""

    step = 0
    for epoch in range(100):  # Large number, will stop based on steps
        print(f"\n🚀 Epoch {epoch + 1}")

        for batch in train_dl:
            if step >= max_steps:
                print(f"✅ Reached max steps ({max_steps})")
                return

            # Training step with automatic KD
            result = training_loop.train_step(batch, step)

            if result is None:
                continue

            step += 1

            # Validation every 500 steps
            if step % 500 == 0:
                val_loss = training_loop.validate(val_dl, step)
                print(f"🎯 Step {step} completed, Val Loss: {val_loss:.4f}")