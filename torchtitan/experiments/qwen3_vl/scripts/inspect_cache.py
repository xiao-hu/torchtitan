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
import logging
import time
from collections import Counter
from pathlib import Path

import torch

from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import \
    PreprocessedVLDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    load_times = []  # Track time to load each sample
    
    # Track tensor dimensions for consistency checking
    tensor_dims = {}  # key -> set of ndims seen
    
    logger.info("Analyzing samples...")
    total_start_time = time.time()
    
    for sample in dataset:
        sample_start_time = time.time()
        sample_count += 1
        
        # Track dimensions for all tensor fields
        for key, val in sample.items():
            if torch.is_tensor(val):
                if key not in tensor_dims:
                    tensor_dims[key] = set()
                tensor_dims[key].add(val.ndim)
        
        # Get sequence length (shape is [batch_size, seq_len], we want seq_len)
        input_ids = sample["input_ids"]
        if input_ids.ndim == 2:
            seq_len = input_ids.shape[1]  # [batch_size, seq_len]
        else:
            seq_len = input_ids.shape[0]  # [seq_len]
        seq_lengths.append(seq_len)
        
        # Get pixel values shape if present
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            pixel_shapes.append(sample["pixel_values"].shape)
        
        # Track load time for this sample
        sample_load_time = time.time() - sample_start_time
        load_times.append(sample_load_time)
        
        # Show progress every 100 samples
        if sample_count % 100 == 0:
            logger.info(f"Processed {sample_count} samples...")
        
        # Limit to first 1000 for quick inspection
        if sample_count >= 1000:
            logger.info(f"Stopping at {sample_count} samples for quick inspection")
            break
    
    total_time = time.time() - total_start_time
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("CACHE INSPECTION RESULTS")
    logger.info("="*60)
    logger.info(f"Total samples analyzed: {sample_count}")
    logger.info(f"Total samples in dataset: {len(dataset.dataset)}")

    # Load metadata if available
    metadata_path = cache_path / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logger.info(f"Metadata reports: {metadata.get('num_packed_samples', 'N/A')} total packed samples")
            logger.info(f"Number of chunks: {metadata.get('num_chunks', 'N/A')}")
    
    logger.info("\n" + "-"*60)
    logger.info("LOADING EFFICIENCY")
    logger.info("-"*60)
    logger.info(f"Total loading time: {total_time:.3f} seconds")
    logger.info(f"Average time per sample: {sum(load_times) / len(load_times) * 1000:.3f} ms")
    logger.info(f"Min time per sample: {min(load_times) * 1000:.3f} ms")
    logger.info(f"Max time per sample: {max(load_times) * 1000:.3f} ms")
    logger.info(f"Median time per sample: {sorted(load_times)[len(load_times)//2] * 1000:.3f} ms")
    logger.info(f"Throughput: {sample_count / total_time:.2f} samples/second")
    
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
    
    # Check dimension consistency
    logger.info("\n" + "-"*60)
    logger.info("DIMENSION CONSISTENCY CHECK")
    logger.info("-"*60)
    
    inconsistent_dims = {key: dims for key, dims in tensor_dims.items() if len(dims) > 1}
    
    if inconsistent_dims:
        logger.warning("⚠️  INCONSISTENT DIMENSIONS DETECTED!")
        logger.warning("Some samples have different number of dimensions for the same field:")
        for key, dims in inconsistent_dims.items():
            logger.warning(f"  {key}: found {sorted(dims)} dimensions across samples")
            logger.warning(f"    → This suggests batch dimension inconsistency!")
    else:
        logger.info("✓ All samples have consistent tensor dimensions")
        for key, dims in sorted(tensor_dims.items()):
            logger.info(f"  {key}: {list(dims)[0]}D tensor (consistent)")
    
    # Check sample structure
    logger.info("\n" + "-"*60)
    logger.info("SAMPLE STRUCTURE (Last Sample)")
    logger.info("-"*60)
    sample_keys = list(sample.keys())
    logger.info(f"Keys in sample: {sample_keys}")
    for key in sample_keys:
        val = sample[key]
        if torch.is_tensor(val):
            logger.info(f"  {key}: tensor shape {val.shape}, dtype {val.dtype}")
            
            # Decode image_grid_thw to show number of images
            if key == "image_grid_thw":
                num_images = val.shape[0] if val.ndim >= 1 else 0
                logger.info(f"    → Contains {num_images} images (each row: [temporal, height, width] grid)")
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
