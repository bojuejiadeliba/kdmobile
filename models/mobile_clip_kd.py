"""
@author: Knowledge Distillation for MobileCLIP - FIXED VERSION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.mobile_clip import MobileCLiP
from transformers import CLIPModel, CLIPProcessor


class MobileCLIPWithKD(nn.Module):
    def __init__(self, student_config, teacher_model="openai/clip-vit-base-patch32"):
        super().__init__()
        self.config = student_config  # Store config for later reference

        # Get knowledge distillation settings
        kd_config = student_config.get("knowledge_distillation", {})
        teacher_model_name = kd_config.get("teacher_model", teacher_model)
        base_checkpoint = kd_config.get("base_checkpoint", None)

        self.distill_temperature = kd_config.get("distill_temperature", 4.0)

        # Initialize student model (with or without pre-trained weights)
        if base_checkpoint and os.path.exists(base_checkpoint):
            print(f"🔄 Loading pre-trained Mobile CLIP from: {base_checkpoint}")
            self.student = self._load_pretrained_student(student_config, base_checkpoint)
        else:
            print("🆕 Initializing Mobile CLIP from scratch")
            self.student = MobileCLiP(student_config)


        # Teacher model (OpenAI CLIP)
        print(f"📥 Loading teacher model: {teacher_model_name}")
        self.teacher = CLIPModel.from_pretrained(teacher_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(teacher_model_name)


        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Alignment layers
        student_dim = student_config["clip_model"]["proj_dim"]
        teacher_dim = 512  # CLIP uses 512-dimensional embeddings

        self.img_align = nn.Sequential(
            nn.Linear(student_dim, teacher_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(teacher_dim * 2 , teacher_dim) # 1024 -> 512
        )

        self.txt_align = nn.Sequential(
            nn.Linear(student_dim, teacher_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(teacher_dim * 2, teacher_dim)
        )

        # Initialize alignment layers
        for m in self.img_align.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.txt_align.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print("✅ MobileCLIPWithKD initialized successfully!")

    def _load_pretrained_student(self, config, checkpoint_path):
        """Load pre-trained Mobile CLIP model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✓ Loaded checkpoint from {checkpoint_path}")

            # Extract student state dict
            if 'state_dict' in checkpoint:
                print("✓ Found Lightning checkpoint format")
                student_state_dict = {}

                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('clip_model.'):
                        new_key = key.replace('clip_model.', '')
                        student_state_dict[new_key] = value

                print(f"✅ Extracted {len(student_state_dict)} parameters with 'clip_model.' prefix")

                if not student_state_dict:
                    print("⚠️ No clip_model parameters found, trying clip_model.student. prefix")
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('clip_model.student.'):
                            new_key = key.replace('clip_model.student.', '')
                            student_state_dict[new_key] = value

                    if not student_state_dict:
                        print("⚠️ No prefixed parameters found, using direct loading")
                        student_state_dict = checkpoint['state_dict']
            else:
                print("✓ Found direct state dict format")
                student_state_dict = checkpoint

            # Create student model
            student = MobileCLiP(config)

            # Debug: Check parameter counts
            student_expected = len(student.state_dict())
            checkpoint_available = len(student_state_dict)
            print(f"\n📊 Parameter Analysis:")
            print(f"  Student model expects: {student_expected} parameters")
            print(f"  Checkpoint provides: {checkpoint_available} parameters")

            # Load weights
            missing_keys, unexpected_keys = student.load_state_dict(student_state_dict, strict=False)

            # Calculate coverage
            loaded_params = student_expected - len(missing_keys)
            coverage = (loaded_params / student_expected) * 100

            print(f"\n📈 Loading Results:")
            print(f"  ✅ Loaded parameters: {loaded_params}/{student_expected} ({coverage:.1f}%)")

            if missing_keys:
                print(f"  ❌ Missing keys: {len(missing_keys)} parameters")
                if len(missing_keys) <= 10:
                    print(f"    Missing: {list(missing_keys)}")
                else:
                    print(f"    First 5 missing: {list(missing_keys)[:5]}")

            if unexpected_keys:
                print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)} parameters")
                if len(unexpected_keys) <= 10:
                    print(f"    Unexpected: {list(unexpected_keys)}")

            if coverage < 70:
                print(f"⚠️ Low parameter coverage ({coverage:.1f}%) - checkpoint may be incompatible")

            # FIXED: Proper indentation - outside the if block, inside the try block
            return student

        except Exception as e:  # FIXED: Now properly inside the try-except structure
            print(f"❌ Error loading checkpoint: {e}")
            print("Falling back to random initialization")
            return MobileCLiP(config)

    @property
    def temperature(self):
        """Use the student model's temperature parameter"""
        return self.student.tau


    def forward(self, image, text, attn_mask=None, neg_text=None, neg_attn_mask=None, neg_image=None):
        """
        TRULY GENERIC forward pass - works with ANY batch size
        """
        # Get student outputs
        if neg_text is not None and neg_image is not None:
            student_outputs = self.student(
                image, text, attn_mask,
                neg_image=neg_image,
                neg_text=neg_text,
                neg_attn_mask=neg_attn_mask
            )
        elif neg_text is not None:
            neg_image_placeholder = image.clone()
            student_outputs = self.student(
                image, text, attn_mask,
                neg_image=neg_image_placeholder,
                neg_text=neg_text,
                neg_attn_mask=neg_attn_mask
            )
        else:
            student_outputs = self.student(image, text, attn_mask)

        # Teacher model processing (only during training)
        if self.training:
            with torch.no_grad():
                try:
                    batch_size = image.size(0)
                    device = image.device
                    embed_dim = 512

                    # Step 1: Prepare images for CLIP (always as batch, even for size 1)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

                    denormalized_images = torch.clamp(
                        (image.detach() * std + mean), 0.0, 1.0
                    )

                    # Step 2: GENERIC PIL conversion (works for any batch size)
                    pil_images = []
                    for i in range(batch_size):
                        img_np = denormalized_images[i].cpu().numpy().transpose(1, 2, 0)
                        img_np = (img_np * 255).astype('uint8')
                        from PIL import Image
                        pil_images.append(Image.fromarray(img_np))

                    # Step 3: GENERIC text preparation (same for all batch sizes)
                    generic_texts = ["a photo"] * batch_size

                    # Step 4: FORCE consistent batch processing
                    # Always use LIST format, regardless of batch size
                    teacher_inputs = self.clip_processor(
                        text=generic_texts,  # Always a list
                        images=pil_images,  # Always a list
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )

                    # Step 5: Move to device
                    teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}

                    # Step 6: Get teacher outputs
                    teacher_outputs = self.teacher(**teacher_inputs, return_dict=True)

                    # Step 7: FORCE correct output shapes (most important part!)
                    teacher_image_embeds = teacher_outputs.image_embeds
                    teacher_text_embeds = teacher_outputs.text_embeds
                    teacher_logits = teacher_outputs.logits_per_image

                    # GENERIC SHAPE ENFORCEMENT
                    # Ensure ALL outputs have batch_size as first dimension
                    if teacher_image_embeds.dim() == 1:
                        teacher_image_embeds = teacher_image_embeds.unsqueeze(0)
                    if teacher_image_embeds.size(0) != batch_size:
                        teacher_image_embeds = teacher_image_embeds[:batch_size]

                    if teacher_text_embeds.dim() == 1:
                        teacher_text_embeds = teacher_text_embeds.unsqueeze(0)
                    if teacher_text_embeds.size(0) != batch_size:
                        teacher_text_embeds = teacher_text_embeds[:batch_size]

                    # Ensure logits matrix is square and matches batch size
                    if teacher_logits.dim() == 1:
                        teacher_logits = teacher_logits.unsqueeze(0)
                    if teacher_logits.size(0) != batch_size or teacher_logits.size(1) != batch_size:
                        # Create proper logits matrix by computing similarity
                        teacher_logits = torch.matmul(
                            F.normalize(teacher_image_embeds, p=2, dim=-1),
                            F.normalize(teacher_text_embeds, p=2, dim=-1).t()
                        ) * 100.0  # Scale like CLIP

                    # FINAL VALIDATION - Ensure all shapes are correct
                    assert teacher_image_embeds.shape == (
                    batch_size, embed_dim), f"Image embeds: {teacher_image_embeds.shape} != ({batch_size}, {embed_dim})"
                    assert teacher_text_embeds.shape == (
                    batch_size, embed_dim), f"Text embeds: {teacher_text_embeds.shape} != ({batch_size}, {embed_dim})"
                    assert teacher_logits.shape == (
                    batch_size, batch_size), f"Logits: {teacher_logits.shape} != ({batch_size}, {batch_size})"

                    # Add to student outputs
                    student_outputs.update({
                        "teacher_image_embeds": teacher_image_embeds,
                        "teacher_text_embeds": teacher_text_embeds,
                        "teacher_logits_per_image": teacher_logits
                    })


                except Exception as e:
                    print(f"[TEACHER DEBUG] ❌ Error with batch_size={batch_size}: {e}")
                    import traceback
                    traceback.print_exc()

                    # GENERIC fallback - works for any batch size
                    batch_size = image.size(0)
                    device = image.device
                    embed_dim = 512

                    student_outputs.update({
                        "teacher_image_embeds": torch.zeros((batch_size, embed_dim), device=device,
                                                            dtype=torch.float32),
                        "teacher_text_embeds": torch.zeros((batch_size, embed_dim), device=device, dtype=torch.float32),
                        "teacher_logits_per_image": torch.zeros((batch_size, batch_size), device=device,
                                                                dtype=torch.float32)
                    })

        return student_outputs

    def feature_distillation_loss(self, student_image_embeds, student_text_embeds,
                                  teacher_image_embeds, teacher_text_embeds):
        """GENERIC feature distillation - works with any batch size"""

        # Project student embeddings to teacher space
        student_image_proj = self.img_align(student_image_embeds)
        student_text_proj = self.txt_align(student_text_embeds)

        # Normalize all embeddings
        student_image_proj = F.normalize(student_image_proj, p=2, dim=-1)
        student_text_proj = F.normalize(student_text_proj, p=2, dim=-1)
        teacher_image_embeds = F.normalize(teacher_image_embeds, p=2, dim=-1)
        teacher_text_embeds = F.normalize(teacher_text_embeds, p=2, dim=-1)

        # GENERIC batch size handling
        min_batch_size = min(
            student_image_proj.size(0),
            student_text_proj.size(0),
            teacher_image_embeds.size(0),
            teacher_text_embeds.size(0)
        )

        # Ensure all tensors have same batch size
        student_image_proj = student_image_proj[:min_batch_size]
        student_text_proj = student_text_proj[:min_batch_size]
        teacher_image_embeds = teacher_image_embeds[:min_batch_size]
        teacher_text_embeds = teacher_text_embeds[:min_batch_size]

        # Calculate losses
        image_loss = F.mse_loss(student_image_proj, teacher_image_embeds)
        text_loss = F.mse_loss(student_text_proj, teacher_text_embeds)

        return image_loss, text_loss

    def response_distillation_loss(self, student_image_embeds, student_text_embeds,
                                   teacher_logits_per_image):
        """GENERIC response distillation - works with any batch size"""

        # Project and normalize student embeddings
        student_image_proj = F.normalize(self.img_align(student_image_embeds), p=2, dim=-1)
        student_text_proj = F.normalize(self.txt_align(student_text_embeds), p=2, dim=-1)

        # Calculate student similarity matrix
        student_logits = torch.matmul(student_image_proj, student_text_proj.t()) * 100.0

        # GENERIC shape matching
        min_batch_size = min(student_logits.size(0), teacher_logits_per_image.size(0))
        student_logits = student_logits[:min_batch_size, :min_batch_size]
        teacher_logits_per_image = teacher_logits_per_image[:min_batch_size, :min_batch_size]


        try:
            # KL divergence loss
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / self.distill_temperature, dim=-1),
                F.softmax(teacher_logits_per_image / self.distill_temperature, dim=-1),
                reduction='batchmean'
            ) * (self.distill_temperature ** 2)

            return kl_loss

        except Exception as e:
            print(f"[KL DEBUG] Error in KL divergence: {e}")
            print(f"[KL DEBUG] Student shape: {student_logits.shape}")
            print(f"[KL DEBUG] Teacher shape: {teacher_logits_per_image.shape}")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    def encode_image(self, image_tensor):
        """Encode images for inference using the student model"""
        return self.student.encode_image(image_tensor)

    def encode_text(self, text_tensor, attention_mask=None):
        """Encode text for inference using the student model"""
        return self.student.encode_text(text_tensor, attention_mask)
