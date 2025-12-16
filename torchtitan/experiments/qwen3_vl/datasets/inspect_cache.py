#!/usr/bin/env python
"""
Inspect preprocessed VL dataset cache.

Loads preprocessed data and analyzes:
- Sample count
- Sequence length distribution
- Pixel values shapes
- Data integrity
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import PreprocessedVLDataset
from torchtitan.tools.logging import logger


def inspect_cache(cache_dir: str):
    """Inspect preprocessed cache."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
    
    logger.info(f"Inspecting cache: {cache_dir}")
    
    # Load dataset
    logger.info("Loading PreprocessedVLDataset...")
    dataset = PreprocessedVLDataset(
        cache_dir=cache_dir,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
    )
    
    # Collect statistics
    seq_lengths = []
    pixel_shapes = []
    sample_count = 0
    
    logger.info("Analyzing samples...")
    for sample in dataset:
        sample_count += 1
        
        # Get sequence length
        seq_len = sample["input_ids"].shape[0]
        seq_lengths.append(seq_len)
        
        # Get pixel values shape if present
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            pixel_shapes.append(sample["pixel_values"].shape)
        
        # Show progress every 100 samples
        if sample_count % 100 == 0:
            logger.info(f"Processed {sample_count} samples...")
        
        # Limit to first 1000 for quick inspection
        if sample_count >= 1000:
            logger.info(f"Stopping at {sample_count} samples for quick inspection")
            break
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("CACHE INSPECTION RESULTS")
    logger.info("="*60)
    logger.info(f"Total samples analyzed: {sample_count}")
    logger.info(f"Metadata reports: {dataset.metadata.get('num_packed_samples', 'N/A')} total packed samples")
    logger.info(f"Number of chunks: {dataset.metadata.get('num_chunks', 'N/A')}")
    
    logger.info("\n" + "-"*60)
    logger.info("SEQUENCE LENGTH DISTRIBUTION")
    logger.info("-"*60)
    logger.info(f"Min sequence length: {min(seq_lengths)}")
    logger.info(f"Max sequence length: {max(seq_lengths)}")
    logger.info(f"Mean sequence length: {sum(seq_lengths) / len(seq_lengths):.1f}")
    logger.info(f"Median sequence length: {sorted(seq_lengths)[len(seq_lengths)//2]}")
    
    # Show distribution buckets
    logger.info("\nSequence length buckets:")
    buckets = {
        "0-1000": sum(1 for x in seq_lengths if x <= 1000),
        "1000-2000": sum(1 for x in seq_lengths if 1000 < x <= 2000),
        "2000-4000": sum(1 for x in seq_lengths if 2000 < x <= 4000),
        "4000-6000": sum(1 for x in seq_lengths if 4000 < x <= 6000),
        "6000-8000": sum(1 for x in seq_lengths if 6000 < x <= 8000),
        "8000+": sum(1 for x in seq_lengths if x > 8000),
    }
    for bucket, count in buckets.items():
        pct = 100.0 * count / len(seq_lengths)
        logger.info(f"  {bucket:12s}: {count:5d} ({pct:5.1f}%)")
    
    if pixel_shapes:
        logger.info("\n" + "-"*60)
        logger.info("PIXEL VALUES DISTRIBUTION")
        logger.info("-"*60)
        shape_counts = Counter(pixel_shapes)
        logger.info(f"Unique pixel_values shapes: {len(shape_counts)}")
        logger.info("Most common shapes:")
        for shape, count in shape_counts.most_common(10):
            pct = 100.0 * count / len(pixel_shapes)
            logger.info(f"  {str(shape):40s}: {count:5d} ({pct:5.1f}%)")
    
    # Check sample structure
    logger.info("\n" + "-"*60)
    logger.info("SAMPLE STRUCTURE")
    logger.info("-"*60)
    sample_keys = list(sample.keys())
    logger.info(f"Keys in sample: {sample_keys}")
    for key in sample_keys:
        val = sample[key]
        if torch.is_tensor(val):
            logger.info(f"  {key}: tensor shape {val.shape}, dtype {val.dtype}")
        else:
            logger.info(f"  {key}: {type(val)}")
    
    logger.info("\n" + "="*60)
    logger.info("Inspection complete!")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Inspect preprocessed VL dataset cache")
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Path to cache directory"
    )
    
    args = parser.parse_args()
    inspect_cache(args.cache_dir)


if __name__ == "__main__":
    main()
