#!/usr/bin/env python3
"""
Verification script for Qwen3-VL dataloader and packing.

This script:
1. Loads a few samples from the VQAv2 dataset
2. Applies format_vqav2_sample to each sample
3. Verifies the output format and correctness
4. Tests HuggingFaceVLDataset with packing enabled
5. Reports any errors or issues

Usage:
    python verify_dataloader.py [--num-samples 10] [--test-packing]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from torchtitan
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from datasets import load_dataset
from transformers import Qwen3VLProcessor

from torchtitan.experiments.qwen3_vl.datasets.utils import (
    collate_vl_batch, preprocess_qwen_visual_pil)
from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import (
    HuggingFaceVLDataset, format_vqav2_sample)


def verify_sample_format(sample_idx: int, original: dict, formatted: dict) -> tuple[bool, str]:
    """
    Verify that a formatted sample has the correct structure.
    
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    # Check required top-level keys (new format uses "messages")
    required_keys = {"messages"}
    if not required_keys.issubset(formatted.keys()):
        missing = required_keys - set(formatted.keys())
        errors.append(f"Missing required keys: {missing}")
    
    # Check messages structure
    if "messages" in formatted:
        messages = formatted["messages"]
        if not isinstance(messages, list):
            errors.append(f"'messages' should be a list, got {type(messages)}")
        elif len(messages) != 2:
            errors.append(f"'messages' should have 2 items, got {len(messages)}")
        else:
            # Check first message (user)
            user_msg = messages[0]
            if user_msg.get("role") != "user":
                errors.append(f"First message should be from 'user', got '{user_msg.get('role')}'")
            
            # Check user content structure
            user_content = user_msg.get("content", [])
            if not isinstance(user_content, list):
                errors.append(f"User content should be a list, got {type(user_content)}")
            else:
                # Should have image and text
                has_image = any(item.get("type") == "image" for item in user_content)
                has_text = any(item.get("type") == "text" for item in user_content)
                if not has_image:
                    errors.append("User content should contain an image")
                if not has_text:
                    errors.append("User content should contain text")
                
                # Check question is in text
                question = original.get("question", "")
                text_parts = [item.get("text", "") for item in user_content if item.get("type") == "text"]
                if question and not any(question in text for text in text_parts):
                    errors.append(f"Question '{question}' not found in user text")
            
            # Check second message (assistant)
            assistant_msg = messages[1]
            if assistant_msg.get("role") != "assistant":
                errors.append(f"Second message should be from 'assistant', got '{assistant_msg.get('role')}'")
            
            answer = assistant_msg.get("content", "")
            # Check that answer is either from original answers or "unknown"
            if answer != "unknown":
                original_answers = original.get("answers", [])
                if original_answers:
                    valid_answers = [a.get("answer", "") for a in original_answers if isinstance(a, dict)]
                    if answer not in valid_answers:
                        errors.append(f"Answer '{answer}' not in original answers: {valid_answers}")
    
    if errors:
        return False, "; ".join(errors)
    return True, "OK"


def test_collator_integration(
    dataset_name: str = "vqav2",
    dataset_path: str = "lmms-lab/VQAv2",
    num_samples: int = 3,
):
    """Test full pipeline including collator wrapper."""
    print("\n" + "=" * 60)
    print("TESTING COLLATOR INTEGRATION")
    print("=" * 60)
    
    try:
        # Load processor (same as training config)
        print("Loading Qwen3VL processor...")
        processor = Qwen3VLProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            trust_remote_code=True
        )
        print("✓ Processor loaded")
        
        # Create dataset
        print("\nCreating dataset...")
        dataset = HuggingFaceVLDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            processor=processor,
            preprocess_fn=preprocess_qwen_visual_pil,
            batch_size=2,
            seq_len=512,
            packing_buffer_size=0,  # No packing for this test
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )
        print("✓ Dataset created")
        
        # Collect samples for collator
        print(f"\nCollecting {num_samples} samples for collator...")
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(sample)
            print(f"  Sample {i+1}: input_ids={sample['input_ids'].shape}, "
                  f"pixel_values={'list' if isinstance(sample.get('pixel_values'), list) else sample['pixel_values'].shape if sample.get('pixel_values') is not None else 'None'}, "
                  f"image_grid_thw={sample['image_grid_thw'].shape if sample.get('image_grid_thw') is not None else 'None'}")
        
        # Test collator
        print(f"\nTesting collator with {len(samples)} samples...")
        try:
            input_dict, labels = collate_vl_batch(samples, processor)
            
            print("\n  Collator output:")
            print("    Type: tuple of (dict, tensor)")
            print(f"    input_dict keys: {list(input_dict.keys())}")
            print(f"    input shape: {input_dict['input'].shape}")
            print(f"    labels shape: {labels.shape}")
            
            # Validate shapes
            assert input_dict['input'].shape == labels.shape, \
                f"Shape mismatch: input={input_dict['input'].shape}, labels={labels.shape}"
            
            # Check pixel_values dimensions
            if 'pixel_values' in input_dict and input_dict['pixel_values'] is not None:
                pv = input_dict['pixel_values']
                if isinstance(pv, list):
                    print(f"    pixel_values: list of {len(pv)} tensors")
                    # Check each tensor is 2D
                    for i, t in enumerate(pv):
                        assert t.dim() == 2, f"pixel_values[{i}] should be 2D, got {t.dim()}D"
                else:
                    print(f"    pixel_values shape: {pv.shape}")
            
            # Check image_grid_thw dimensions
            if 'image_grid_thw' in input_dict and input_dict['image_grid_thw'] is not None:
                igt = input_dict['image_grid_thw']
                print(f"    image_grid_thw shape: {igt.shape}")
                if isinstance(igt, list):
                    print(f"      (list of {len(igt)} tensors)")
                    for i, t in enumerate(igt):
                        assert t.dim() == 1 and t.shape[0] == 3, \
                            f"image_grid_thw[{i}] should be [3], got {t.shape}"
                else:
                    assert igt.dim() == 2 and igt.shape[1] == 3, \
                        f"image_grid_thw should be [N, 3], got {igt.shape}"
            
            print("\n  ✓ Collator test passed")
            print("\n" + "=" * 60)
            print("COLLATOR TEST PASSED")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n  ✗ Collator test failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 60)
            print("COLLATOR TEST FAILED")
            print("=" * 60)
            return False
        
    except Exception as e:
        print(f"\n✗ COLLATOR TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("COLLATOR TEST FAILED")
        print("=" * 60)
        return False


def test_dataloader_with_packing(
    dataset_name: str = "vqav2",
    dataset_path: str = "lmms-lab/VQAv2", 
    num_samples: int = 50,  # More samples to catch edge cases
    packing_buffer_size: int = 100,
):
    """Test HuggingFaceVLDataset with sample packing AND collator (FULL TRAINING PIPELINE)."""
    print("\n" + "=" * 60)
    print("TESTING FULL TRAINING PIPELINE: PACKING + COLLATOR")
    print("=" * 60)
    
    try:
        # Load processor (same as training config)
        print("Loading Qwen3VL processor...")
        processor = Qwen3VLProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            trust_remote_code=True
        )
        print("✓ Processor loaded")
        
        # Create dataset with packing - EXACT training config
        print(f"\nCreating HuggingFaceVLDataset with packing_buffer_size={packing_buffer_size}...")
        print("  (Using batch_size=1, seq_len=4096 to EXACTLY match training config)")
        dataset = HuggingFaceVLDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            processor=processor,
            preprocess_fn=preprocess_qwen_visual_pil,
            batch_size=1,  # EXACT match: local_batch_size=1 in config
            seq_len=4096,  # EXACT match: seq_len=4096 in config
            packing_buffer_size=packing_buffer_size,
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )
        print("✓ Dataset created")
        
        # CRITICAL: Now test collator with packed samples (THIS IS WHAT TRAINING DOES!)
        print("\n🔥 CRITICAL TEST: Passing packed samples through collator...")
        print("   (This is exactly what happens in training loop)")
        
        sample_count = 0
        error_count = 0
        max_consecutive_errors = 5  # Stop after 5 consecutive errors
        
        for i, packed_sample in enumerate(dataset):
            if i >= num_samples:
                break
            
            if error_count >= max_consecutive_errors:
                print(f"\n✗ Stopping: {max_consecutive_errors} consecutive errors detected")
                break
                
            sample_count += 1
            
            # Debug: Check what packer returns
            if i == 0:
                print("\n  📊 First packed sample from dataset:")
                print(f"     Type: {type(packed_sample)}")
                print(f"     Keys: {list(packed_sample.keys())}")
                if "pixel_values" in packed_sample and packed_sample["pixel_values"] is not None:
                    pv = packed_sample["pixel_values"]
                    if isinstance(pv, list):
                        print(f"     pixel_values: LIST of {len(pv)} items")
                        if len(pv) > 0:
                            print(f"       First item type: {type(pv[0])}")
                            if torch.is_tensor(pv[0]):
                                print(f"       First item shape: {pv[0].shape}")
                    else:
                        print(f"     pixel_values shape: {pv.shape}")
                if "image_grid_thw" in packed_sample and packed_sample["image_grid_thw"] is not None:
                    igt = packed_sample["image_grid_thw"]
                    if isinstance(igt, list):
                        print(f"     image_grid_thw: LIST of {len(igt)} items")
                        if len(igt) > 0:
                            print(f"       First item type: {type(igt[0])}")
                            if torch.is_tensor(igt[0]):
                                print(f"       First item shape: {igt[0].shape}")
                    else:
                        print(f"     image_grid_thw shape: {igt.shape}")
            
            # 🔥 THIS IS THE CRITICAL TEST: Pass through collator!
            try:
                # Wrap in list as training does (batch_size=1)
                input_dict, labels = collate_vl_batch([packed_sample], processor)
                
                if i < 3:  # Print first 3 for debugging
                    print(f"\n  ✓ Sample {i+1}: Collator succeeded")
                    print(f"     input shape: {input_dict['input'].shape}")
                    print(f"     labels shape: {labels.shape}")
                    if "pixel_values" in input_dict and input_dict["pixel_values"] is not None:
                        print(f"     pixel_values shape: {input_dict['pixel_values'].shape}")
                    if "image_grid_thw" in input_dict and input_dict["image_grid_thw"] is not None:
                        print(f"     image_grid_thw shape: {input_dict['image_grid_thw'].shape}")
                
                # Reset error count on success
                error_count = 0
                
            except Exception as e:
                print(f"\n  ✗ Sample {i+1}: COLLATOR FAILED!")
                print(f"     Error: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                
                # This is the bug we're looking for!
                if "expected Tensor" in str(e) and "but got list" in str(e):
                    print("\n  🎯 FOUND THE BUG! This is the exact error from training!")
                    print("     The packer returns pixel_values as a LIST,")
                    print("     but the collator expects a TENSOR!")
                    return False
                
        if error_count >= max_consecutive_errors:
            print(f"\n✗ Failed: {max_consecutive_errors} consecutive errors")
            print("\n" + "=" * 60)
            print("FULL PIPELINE TEST FAILED")
            print("=" * 60)
            return False
        else:
            print(f"\n✓ Successfully processed {sample_count} packed samples through collator!")
            print("=" * 60)
            print("FULL PIPELINE TEST PASSED")
            print("=" * 60)
            return True
        
    except Exception as e:
        print(f"\n✗ PACKING TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("PACKING TEST FAILED")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Qwen3-VL dataloader and packing")
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
    parser.add_argument(
        "--test-packing",
        action="store_true",
        help="Test HuggingFaceVLDataset with packing enabled"
    )
    parser.add_argument(
        "--packing-buffer-size",
        type=int,
        default=100,
        help="Buffer size for packing test (default: 100)"
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
                    print(f"  Formatted answer: {formatted['messages'][1]['content']}")
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
        print("FORMAT VALIDATION SUMMARY:")
        print(f"  Total samples tested: {args.num_samples}")
        print(f"  ✓ Successful: {success_count}")
        print(f"  ✗ Errors: {error_count}")
        print("=" * 60)
        
        format_test_passed = (error_count == 0)
        
        if format_test_passed:
            print()
            print("🎉 All samples processed successfully!")
            print("The format_vqav2_sample fix is working correctly.")
        else:
            print()
            print("⚠️  Some samples had errors. Please review the output above.")
        
        # Run collator test
        collator_test_passed = test_collator_integration(
            dataset_name="vqav2",
            dataset_path=args.dataset,
            num_samples=3,
        )
        
        # Run packing test if requested
        packing_test_passed = True
        if args.test_packing:
            print("\n")
            packing_test_passed = test_dataloader_with_packing(
                dataset_name="vqav2",
                dataset_path=args.dataset,
                num_samples=5,
                packing_buffer_size=args.packing_buffer_size,
            )
        
        # Overall result
        print("\n" + "=" * 60)
        print("OVERALL RESULT:")
        print(f"  Format test: {'✓ PASSED' if format_test_passed else '✗ FAILED'}")
        print(f"  Collator test: {'✓ PASSED' if collator_test_passed else '✗ FAILED'}")
        if args.test_packing:
            print(f"  Packing test: {'✓ PASSED' if packing_test_passed else '✗ FAILED'}")
        print("=" * 60)
        
        if format_test_passed and collator_test_passed and packing_test_passed:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please review the output above.")
            return 1
            
    except Exception as e:
        print(f"✗ Failed to load dataset: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
