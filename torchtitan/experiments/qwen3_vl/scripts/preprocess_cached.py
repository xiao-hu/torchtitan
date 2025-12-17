# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline preprocessing script for Qwen3-VL datasets.

Preprocesses and packs samples offline to eliminate runtime CPU bottleneck.
Saves preprocessed data to cache for fast loading during training.

This script leverages the existing HuggingFaceVLDataset infrastructure
to ensure consistency between online and offline preprocessing.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tensordict import TensorDict
from tqdm import tqdm
from transformers import Qwen3VLProcessor

from torchtitan.experiments.qwen3_vl.datasets.cached_vl_datasets import \
    get_cache_key
from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import \
    HuggingFaceVLDataset
from torchtitan.experiments.qwen3_vl.train_spec import \
    preprocess_qwen_visual_pil
from torchtitan.tools.logging import logger


def extract_tensordict_from_sample(sample: Dict[str, Any]) -> TensorDict:
    """
    Extract TensorDict from BatchFeature sample.
    
    TensorDict natively supports PyTorch tensors with efficient storage.
    No conversion needed - direct tensor storage!
    
    Args:
        sample: BatchFeature or dict containing tensors
        
    Returns:
        TensorDict with all tensor fields
    """
    # Extract base dict
    if hasattr(sample, 'data'):
        data = dict(sample.data)
    elif isinstance(sample, dict):
        data = sample
    else:
        data = dict(sample)
    
    # Create TensorDict (handles tensors natively)
    return TensorDict(data, batch_size=[])


def get_cache_dir(
    base_cache_dir: str,
    dataset_name: str,
    seq_len: int,
    buffer_size: int,
    model_path: str,
) -> Path:
    """Get cache directory path."""
    cache_key = get_cache_key(dataset_name, seq_len, buffer_size, model_path)
    return Path(base_cache_dir) / cache_key


def preprocess_and_cache(
    dataset_name: str,
    model_path: str,
    seq_len: int,
    buffer_size: int,
    batch_size: int,
    cache_dir: Path,
    force: bool = False,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Preprocess dataset and save to cache using HuggingFaceVLDataset.
    
    Args:
        dataset_name: Dataset name from VL_DATASETS (e.g., "vqav2", "vqav2_validation")
        model_path: Path to model for processor
        seq_len: Maximum sequence length
        buffer_size: Packing buffer size
        batch_size: Batch size for packing
        cache_dir: Directory to save cache
        force: Force reprocessing even if cache exists
        max_samples: Maximum samples to process (None = all)
    
    Returns:
        Dictionary with cache metadata
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "packed_samples.pt"
    metadata_file = cache_dir / "metadata.json"
    
    # Check if cache exists
    if not force and cache_file.exists():
        logger.info(f"Cache already exists at {cache_dir}. Use --force to regenerate.")
        with open(metadata_file) as f:
            return json.load(f)
    
    logger.info(f"Preprocessing {dataset_name} - this may take 2-6 hours...")
    
    # Load processor
    logger.info(f"Loading processor from {model_path}")
    processor = Qwen3VLProcessor.from_pretrained(model_path)
    
    # Create HuggingFaceVLDataset with packing enabled
    logger.info(f"Creating HuggingFaceVLDataset for {dataset_name}")
    vl_dataset = HuggingFaceVLDataset(
        dataset_name=dataset_name,
        dataset_path=None,  # Use default from registry
        processor=processor,
        preprocess_fn=preprocess_qwen_visual_pil,
        batch_size=batch_size,
        seq_len=seq_len,
        packing_buffer_size=buffer_size,
        dp_rank=0,  # Single process for preprocessing
        dp_world_size=1,
        infinite=False,  # Single pass through dataset
    )
    
    # Process samples incrementally - save each individually (variable lengths!)
    sample_count = 0
    error_count = 0
    
    # Create samples directory
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Calculate total for progress bar
    total_samples = max_samples if max_samples else len(vl_dataset._data)
    logger.info(f"Processing {total_samples} samples with packing (buffer_size={buffer_size})...")
    logger.info("Saving each sample individually (variable lengths preserved)")
    logger.info("Output: LazyStackedTensorDict (memory-mapped, instant loading, native tensors)")
    
    try:
        for sample in tqdm(vl_dataset, desc="Preprocessing", total=total_samples, unit="samples"):
            # Extract TensorDict (no conversion needed!)
            td = extract_tensordict_from_sample(sample)
            sample_count += 1
            
            # Save each sample individually (variable lengths!)
            sample_path = samples_dir / f"sample_{sample_count:06d}"
            td.memmap_(sample_path)
            
            # Stop if max_samples reached
            if max_samples and sample_count >= max_samples:
                logger.info(f"Reached max_samples={max_samples}, stopping")
                break
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
    
    error_count = vl_dataset.error_count
    
    
    # Get packing stats
    packing_stats = {}
    if hasattr(vl_dataset, 'packer'):
        packing_stats = vl_dataset.packer.get_stats()
    
    logger.info(f"✓ Saved {sample_count} samples individually")
    logger.info(f"  Samples in: {samples_dir}")
    logger.info("  Use LazyStackedTensorDict at load time for seamless access")
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "buffer_size": buffer_size,
        "num_samples": sample_count,
        "error_count": error_count,
        "format": "lazystacked_tensordict",
        "packing_efficiency": packing_stats.get('avg_samples_per_pack', 0),
    }
    
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save packing stats
    with open(cache_dir / "packing_stats.json", "w") as f:
        json.dump(packing_stats, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Preprocessing complete!")
    logger.info("="*60)
    logger.info(f"Packed samples: {sample_count}")
    logger.info(f"Packing efficiency: {packing_stats.get('avg_samples_per_pack', 0):.2f} samples/pack")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Samples location: {samples_dir}")
    logger.info("Format: LazyStackedTensorDict (memory-mapped, instant loading, native tensors)")
    logger.info("="*60)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and cache VL dataset for faster training"
    )
    parser.add_argument(
        "--dataset-name",
        default="vqav2",
        help="Dataset name from VL_DATASETS (e.g., 'vqav2', 'vqav2_validation')"
    )
    parser.add_argument(
        "--model-path",
        default="/checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Path to model for processor",
    )
    parser.add_argument("--seq-len", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--buffer-size", type=int, default=75, help="Packing buffer size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--cache-dir",
        default="/checkpoints/xxie-sandbox/preprocessed_cache",
        help="Base cache directory",
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--max-samples", type=int, help="Max samples (for testing)")
    
    args = parser.parse_args()
    
    cache_dir = get_cache_dir(
        args.cache_dir,
        args.dataset_name,
        args.seq_len,
        args.buffer_size,
        args.model_path,
    )
    
    preprocess_and_cache(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        seq_len=args.seq_len,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        force=args.force,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
