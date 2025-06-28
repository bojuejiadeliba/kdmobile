#!/usr/bin/env python3
"""
Checkpoint Inspector - Analyze the structure of saved model checkpoints
"""
import torch
import argparse
from collections import defaultdict


def inspect_checkpoint(checkpoint_path):
    """Inspect a PyTorch checkpoint file"""
    print(f"\n🔍 INSPECTING CHECKPOINT: {checkpoint_path}")
    print("=" * 80)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Successfully loaded checkpoint")

        # Top-level structure
        print(f"\n📋 TOP-LEVEL KEYS:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  📁 {key}: {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  📊 {key}: tensor {list(checkpoint[key].shape)}")
            else:
                print(f"  📄 {key}: {type(checkpoint[key])}")

        # Focus on state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n🎯 STATE_DICT ANALYSIS:")
            print(f"  Total parameters: {len(state_dict)}")

            # Analyze prefixes
            prefixes = defaultdict(int)
            sample_keys = {}

            for key in state_dict.keys():
                # Extract prefix (everything before the first dot after a potential module name)
                parts = key.split('.')
                if len(parts) >= 2:
                    prefix = f"{parts[0]}.{parts[1]}" if len(parts) > 2 else parts[0]
                else:
                    prefix = parts[0]

                prefixes[prefix] += 1

                # Store sample keys for each prefix
                if prefix not in sample_keys:
                    sample_keys[prefix] = []
                if len(sample_keys[prefix]) < 3:
                    sample_keys[prefix].append(key)

            print(f"\n📊 PREFIX ANALYSIS:")
            for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
                print(f"  🏷️  {prefix}: {count} parameters")
                print(f"      Examples: {sample_keys[prefix][:2]}")
                if len(sample_keys[prefix]) > 2:
                    print(f"      ...")
                print()

            # Look for specific patterns we care about
            patterns = {
                'clip_model.': [],
                'clip_model.student.': [],
                'model.': [],
                'student.': [],
                'tau': [],
                'img_model': [],
                'text_model': [],
                'projection': []
            }

            for key in state_dict.keys():
                for pattern in patterns:
                    if pattern in key:
                        patterns[pattern].append(key)

            print(f"🔍 PATTERN SEARCH RESULTS:")
            for pattern, matches in patterns.items():
                if matches:
                    print(f"  ✅ '{pattern}': {len(matches)} matches")
                    print(f"      First few: {matches[:3]}")
                    if len(matches) > 3:
                        print(f"      ...")
                else:
                    print(f"  ❌ '{pattern}': No matches")
                print()

            # Sample some actual parameter names
            print(f"📝 SAMPLE PARAMETER NAMES (first 20):")
            for i, key in enumerate(list(state_dict.keys())[:20]):
                tensor_shape = list(state_dict[key].shape) if hasattr(state_dict[key], 'shape') else 'scalar'
                print(f"  {i + 1:2d}. {key}")
                print(f"      Shape: {tensor_shape}")

            if len(state_dict) > 20:
                print(f"  ... and {len(state_dict) - 20} more parameters")

        else:
            print("❌ No 'state_dict' found in checkpoint")

        # Check if it's a Lightning checkpoint
        if 'epoch' in checkpoint:
            print(f"\n⚡ LIGHTNING CHECKPOINT INFO:")
            print(f"  Epoch: {checkpoint['epoch']}")
            if 'global_step' in checkpoint:
                print(f"  Global step: {checkpoint['global_step']}")
            if 'pytorch-lightning_version' in checkpoint:
                print(f"  PyTorch Lightning version: {checkpoint['pytorch-lightning_version']}")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

    return checkpoint


def suggest_loading_strategy(checkpoint_path):
    """Suggest the best loading strategy based on checkpoint structure"""
    checkpoint = inspect_checkpoint(checkpoint_path)
    if not checkpoint or 'state_dict' not in checkpoint:
        return

    state_dict = checkpoint['state_dict']

    print(f"\n💡 SUGGESTED LOADING STRATEGY:")
    print("=" * 50)

    # Count different prefix patterns
    clip_model_count = len([k for k in state_dict.keys() if k.startswith('clip_model.')])
    clip_model_student_count = len([k for k in state_dict.keys() if k.startswith('clip_model.student.')])
    model_count = len([k for k in state_dict.keys() if k.startswith('model.')])
    no_prefix_count = len([k for k in state_dict.keys() if
                           '.' not in k or not any(k.startswith(p) for p in ['clip_model.', 'model.', 'student.'])])

    print(f"📊 Prefix distribution:")
    print(f"  clip_model.*: {clip_model_count}")
    print(f"  clip_model.student.*: {clip_model_student_count}")
    print(f"  model.*: {model_count}")
    print(f"  no clear prefix: {no_prefix_count}")

    # Suggest strategy
    if clip_model_student_count > 0:
        print(f"\n✅ RECOMMENDED: Remove 'clip_model.student.' prefix")
        print(f"   Use: key.replace('clip_model.student.', '')")
    elif clip_model_count > 0:
        print(f"\n✅ RECOMMENDED: Remove 'clip_model.' prefix")
        print(f"   Use: key.replace('clip_model.', '')")
    elif model_count > 0:
        print(f"\n✅ RECOMMENDED: Remove 'model.' prefix")
        print(f"   Use: key.replace('model.', '')")
    else:
        print(f"\n✅ RECOMMENDED: Use direct loading (no prefix removal)")

    # Generate corrected loading function
    print(f"\n🔧 CORRECTED LOADING FUNCTION:")
    print("-" * 40)

    if clip_model_student_count > 0:
        suggested_code = '''
# Remove 'clip_model.student.' prefix
for key, value in checkpoint['state_dict'].items():
    if key.startswith('clip_model.student.'):
        new_key = key.replace('clip_model.student.', '')
        student_state_dict[new_key] = value
'''
    elif clip_model_count > 0:
        suggested_code = '''
# Remove 'clip_model.' prefix  
for key, value in checkpoint['state_dict'].items():
    if key.startswith('clip_model.'):
        new_key = key.replace('clip_model.', '')
        student_state_dict[new_key] = value
'''
    else:
        suggested_code = '''
# Direct loading (no prefix)
student_state_dict = checkpoint['state_dict']
'''

    print(suggested_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint structure")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    args = parser.parse_args()

    suggest_loading_strategy(args.checkpoint_path)