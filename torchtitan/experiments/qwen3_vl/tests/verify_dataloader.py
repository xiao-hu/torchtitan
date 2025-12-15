#!/usr/bin/env python3
"""
Verification script for Qwen3-VL dataloader fix.

This script:
1. Loads a few samples from the VQAv2 dataset
2. Applies format_vqav2_sample to each sample
3. Verifies the output format and correctness
4. Reports any errors or issues

Usage:
    python verify_dataloader.py [--num-samples 10]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from torchtitan
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset

from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import \
    format_vqav2_sample


def verify_sample_format(sample_idx: int, original: dict, formatted: dict) -> tuple[bool, str]:
    """
    Verify that a formatted sample has the correct structure.
    
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = {"conversations", "image", "data_path"}
    if not required_keys.issubset(formatted.keys()):
        missing = required_keys - set(formatted.keys())
        errors.append(f"Missing required keys: {missing}")
    
    # Check conversations structure
    if "conversations" in formatted:
        convos = formatted["conversations"]
        if not isinstance(convos, list):
            errors.append(f"'conversations' should be a list, got {type(convos)}")
        elif len(convos) != 2:
            errors.append(f"'conversations' should have 2 items, got {len(convos)}")
        else:
            # Check first message (human)
            if convos[0].get("from") != "human":
                errors.append(f"First message should be from 'human', got '{convos[0].get('from')}'")
            if "<image>" not in convos[0].get("value", ""):
                errors.append("First message should contain '<image>' token")
            if original.get("question", "") not in convos[0].get("value", ""):
                errors.append("First message should contain the question")
            
            # Check second message (gpt)
            if convos[1].get("from") != "gpt":
                errors.append(f"Second message should be from 'gpt', got '{convos[1].get('from')}'")
            
            answer = convos[1].get("value", "")
            # Check that answer is either from original answers or "unknown"
            if answer != "unknown":
                original_answers = original.get("answers", [])
                if original_answers:
                    valid_answers = [a.get("answer", "") for a in original_answers if isinstance(a, dict)]
                    if answer not in valid_answers:
                        errors.append(f"Answer '{answer}' not in original answers: {valid_answers}")
    
    # Check image structure
    if "image" in formatted:
        images = formatted["image"]
        if not isinstance(images, list):
            errors.append(f"'image' should be a list, got {type(images)}")
        elif len(images) != 1:
            errors.append(f"'image' should have 1 item, got {len(images)}")
    
    # Check data_path
    if "data_path" in formatted:
        if formatted["data_path"] != "":
            errors.append(f"'data_path' should be empty string, got '{formatted['data_path']}'")
    
    if errors:
        return False, "; ".join(errors)
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Verify Qwen3-VL dataloader fix")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to verify (default: 10)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmms-lab/VQAv2",
        help="Dataset path (default: lmms-lab/VQAv2)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)"
    )
    args = parser.parse_args()
    
    print(f"Loading {args.num_samples} samples from {args.dataset} ({args.split} split)...")
    print()
    
    try:
        # Load dataset
        dataset = load_dataset(args.dataset, split=args.split, streaming=False)
        print(f"✓ Successfully loaded dataset: {len(dataset)} total samples")
        print()
        
        # Test format_vqav2_sample on multiple samples
        success_count = 0
        error_count = 0
        
        for i, sample in enumerate(dataset.select(range(min(args.num_samples, len(dataset))))):
            print(f"Sample {i+1}/{args.num_samples}:")
            print(f"  Question: {sample['question']}")
            
            # Show original answers format
            answers = sample.get("answers", [])
            if answers and isinstance(answers, list) and isinstance(answers[0], dict):
                answer_texts = [a.get("answer", "N/A") for a in answers[:3]]  # Show first 3
                print(f"  Answers: {answer_texts}")
            
            try:
                # Apply formatter
                formatted = format_vqav2_sample(sample)
                
                # Verify format
                is_valid, message = verify_sample_format(i, sample, formatted)
                
                if is_valid:
                    print(f"  ✓ {message}")
                    print(f"  Formatted answer: {formatted['conversations'][1]['value']}")
                    success_count += 1
                else:
                    print(f"  ✗ VALIDATION ERROR: {message}")
                    error_count += 1
                    
            except Exception as e:
                print(f"  ✗ EXCEPTION: {type(e).__name__}: {e}")
                error_count += 1
                
                # Show traceback for first error
                if error_count == 1:
                    import traceback
                    print()
                    print("Full traceback for first error:")
                    traceback.print_exc()
                    print()
            
            print()
        
        # Summary
        print("=" * 60)
        print("SUMMARY:")
        print(f"  Total samples tested: {args.num_samples}")
        print(f"  ✓ Successful: {success_count}")
        print(f"  ✗ Errors: {error_count}")
        print("=" * 60)
        
        if error_count == 0:
            print()
            print("🎉 All samples processed successfully!")
            print("The dataloader fix is working correctly.")
            return 0
        else:
            print()
            print("⚠️  Some samples had errors. Please review the output above.")
            return 1
            
    except Exception as e:
        print(f"✗ Failed to load dataset: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
