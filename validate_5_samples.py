"""
Debug script that follows the EXACT same flow as evaluate_models.py
to test if the model is working properly (not stuck at 1.386 baseline)
"""

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from trainer import LitMobileCLiP
from transformers import CLIPTokenizerFast
import torchvision
import pickle
from utility.transform_data import IMAGENET_COLOR_MEAN, IMAGENET_COLOR_STD

def text_process(txt: str, tokenizer, max_length: int) -> torch.Tensor:
    """Same as evaluate_models.py"""
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

def get_text_tensors_debug(text_captions, model):
    """Same as evaluate_models.py but with debug info"""
    print("🔍 Processing text captions...")
    text_tensors = []

    for i, caption in enumerate(text_captions):
        txt = caption["input_ids"].to("cuda:0")
        attn_mask = caption["attention_mask"].float().to("cuda:0")

        # Get text encoding (same as evaluation)
        text_encoding = model.encode_text(txt, attn_mask)
        normalized_encoding = F.normalize(text_encoding, p=2, dim=-1)

        # Debug first few
        if i < 3:
            print(f"  Caption {i}: '{text_captions[i]}'")
            print(f"    Raw encoding mean: {text_encoding.mean().item():.6f}")
            print(f"    Raw encoding std: {text_encoding.std().item():.6f}")
            print(f"    Normalized norm: {torch.norm(normalized_encoding).item():.6f}")

        text_tensors.append(normalized_encoding.detach().cpu())

    concat_text_tensor = torch.cat(text_tensors, dim=0)
    print(f"✅ Text tensors shape: {concat_text_tensor.shape}")
    return concat_text_tensor

def debug_image_text_similarity(model, text_tensors, sample_images, class_names):
    """Debug the similarity calculation just like evaluation"""
    print("\n🔍 Testing image-text similarity calculation...")

    # Take first few images for debugging
    num_debug_samples = 5

    with torch.no_grad():
        for i in range(min(num_debug_samples, len(sample_images))):
            img, true_label = sample_images[i]

            # Encode image (same as evaluation)
            img_encoding = model.encode_image(img.unsqueeze(0).to("cuda:0"))
            img_encoding = F.normalize(img_encoding, p=2, dim=-1)

            # Calculate similarities (same as evaluation)
            text_tensors_cuda = text_tensors.cuda()
            similarities = (100.0 * img_encoding @ text_tensors_cuda.t()).softmax(dim=-1)
            pred_label = torch.argmax(similarities, dim=1)

            # Debug output
            print(f"\n  Sample {i+1}:")
            print(f"    True class: {class_names[true_label]} (label {true_label})")
            print(f"    Predicted: {class_names[pred_label.item()]} (label {pred_label.item()})")
            print(f"    Raw similarities (top 3):")

            # Show top 3 similarities
            top_sims, top_indices = torch.topk(similarities[0], 3)
            for j, (sim, idx) in enumerate(zip(top_sims, top_indices)):
                print(f"      {j+1}. {class_names[idx]}: {sim.item():.4f}")

            # Check if similarities are meaningful
            sim_range = similarities.max().item() - similarities.min().item()
            print(f"    Similarity range: {sim_range:.6f}")

            if sim_range < 0.001:
                print("    ❌ 🚨 SIMILARITIES TOO UNIFORM - MODEL NOT WORKING!")
            else:
                print("    ✅ Similarities show variation")

def test_baseline_problem_evaluation_style(model_checkpoint, config_path):
    """Test using the exact same setup as evaluate_models.py"""
    print("🚀 DEBUGGING MODEL USING EVALUATION FLOW")
    print("="*60)

    # Load model and config (same as evaluate_models.py)
    print("📁 Loading model and config...")
    with open(config_path, "r") as fp:
        cfg = yaml.safe_load(fp)

    model = LitMobileCLiP.load_from_checkpoint(model_checkpoint, config=cfg)
    model.freeze()  # Same as evaluation

    # Create tokenizer (same as evaluate_models.py)
    text_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # Create transformations (same as evaluate_models.py)
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD
        ),
    ])

    # Use CIFAR-10 setup (same as evaluate_models.py)
    print("📊 Setting up CIFAR-10 data...")
    prompt_template = "a photo of a"

    # CIFAR-10 labels (same as evaluate_models.py)
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Create prompts (same as evaluate_models.py)
    prompts = list(map(lambda x: prompt_template + " " + x, class_names))
    print("📝 Text prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i}: {prompt}")

    # Tokenize texts (same as evaluate_models.py)
    tokenized_txts = [
        text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
        for txt in prompts
    ]

    # Get text tensors with debug info
    txt_tensors = get_text_tensors_debug(tokenized_txts, model)

    # Load a few CIFAR-10 samples for testing
    print("🖼️  Loading CIFAR-10 samples...")
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root="./eval_datasets", train=False, download=True, transform=transformations
    )

    # Test with first 10 samples
    sample_images = [cifar10_dataset[i] for i in range(10)]

    # Debug similarity calculations
    debug_image_text_similarity(model, txt_tensors, sample_images, class_names)

    # Test for 1.386 baseline problem using the actual model forward pass
    print("\n🎯 TESTING FOR 1.386 BASELINE PROBLEM")
    print("-" * 40)

    # Create dummy data similar to training
    dummy_img = torch.randn(2, 3, 224, 224).cuda()
    dummy_txt = torch.randint(0, 1000, (2, 77)).cuda()
    dummy_mask = torch.ones(2, 77).cuda()

    model.train()  # Set to training mode for loss calculation
    losses = []

    with torch.no_grad():
        for i in range(5):
            # Use the same forward pass as training
            outputs = model(
                dummy_img, dummy_txt, dummy_mask,
                neg_image=dummy_img, neg_text=dummy_txt, neg_attn_mask=dummy_mask
            )

            # Calculate loss the same way as training
            loss = model._compute_loss(outputs)
            losses.append(loss.item())

            print(f"  Test {i+1}: Loss = {loss.item():.6f}")

            if i == 0:  # Debug first iteration
                print(f"    Ej: {outputs['Ej'].item():.6f}")
                print(f"    Em: {outputs['Em'].item():.6f}")
                print(f"    Em - Ej: {(outputs['Em'] - outputs['Ej']).item():.6f}")

    # Analyze results
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)

    print(f"\n📊 LOSS ANALYSIS:")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Loss Std Dev: {std_loss:.6f}")
    print(f"Loss Range: {min(losses):.6f} to {max(losses):.6f}")

    # Final verdict
    print("\n🏁 FINAL VERDICT:")
    if abs(avg_loss - 1.386) < 0.05:
        print("❌ 🚨 MODEL IS STUCK AT RANDOM BASELINE!")
        print("   Loss ≈ 1.386 indicates ln(2) = random contrastive loss")
        print("   This means the model hasn't learned anything.")
        print("   DO NOT use this as baseline for KD comparison!")
        return False
    elif std_loss < 0.001:
        print("❌ 🚨 MODEL OUTPUT IS TOO CONSISTENT!")
        print("   This might indicate frozen or broken parameters.")
        return False
    else:
        print("✅ ✨ MODEL APPEARS TO BE WORKING!")
        print(f"   Loss ({avg_loss:.3f}) is different from random baseline (1.386)")
        print("   Model shows learning and can be used as baseline.")
        return True

if __name__ == "__main__":
    # Update these paths
    model_checkpoint = "D:/AU/Dissertation/KLMobileProject/checkpoints/clip_mobile_test.ckpt"
    config_path = "D:/AU/Dissertation/KLMobileProject/configs/config.yaml"

    test_baseline_problem_evaluation_style(model_checkpoint, config_path)