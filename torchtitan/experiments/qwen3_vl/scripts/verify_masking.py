#!/usr/bin/env python3
"""Test to verify the masking logic correctly masks only assistant turns."""

import numpy as np
# Create a test sample with user question and assistant answer
from PIL import Image
from transformers import Qwen3VLProcessor

from torchtitan.experiments.qwen3_vl.train_spec import \
    preprocess_qwen_visual_pil

# Load processor
processor = Qwen3VLProcessor.from_pretrained(
    '/checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct', 
    trust_remote_code=True
)

print('Token IDs:')
print(f'<|im_start|>: {processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)}')
print(f'user: {processor.tokenizer.encode("user", add_special_tokens=False)}')
print(f'assistant: {processor.tokenizer.encode("assistant", add_special_tokens=False)}')
print(f'<|im_end|>: {processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)}')

# Create dummy image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

test_sample = {
    "conversations": [
        {"from": "human", "value": "<image>What color is this?"},
        {"from": "gpt", "value": "Red"}
    ],
    "image": [img],
    "data_path": ""
}

# Process the sample
result = preprocess_qwen_visual_pil([test_sample], processor)

# Analyze the labels
input_ids = result["input_ids"][0].tolist()
labels = result["labels"][0].tolist()

print("=" * 80)
print("MASKING VERIFICATION TEST")
print("=" * 80)

# Decode tokens to see what's masked vs unmasked
IGNORE_INDEX = -100

print("\nToken sequence analysis:")
print("-" * 80)

for i, (token_id, label) in enumerate(zip(input_ids, labels)):
    token_text = processor.tokenizer.decode([token_id])
    is_masked = (label == IGNORE_INDEX)
    status = "MASKED  " if is_masked else "UNMASKED"
    print(f"{i:4d}: {status} | Token ID: {token_id:6d} | Text: {repr(token_text)}")

# Count masked vs unmasked tokens
num_masked = sum(1 for lb in labels if lb == IGNORE_INDEX)
num_unmasked = len(labels) - num_masked

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total tokens: {len(labels)}")
print(f"Masked (user turn): {num_masked}")
print(f"Unmasked (assistant turn): {num_unmasked}")

# Verify the unmasked part contains "Red"
unmasked_tokens = [token_id for token_id, label in zip(input_ids, labels) if label != IGNORE_INDEX]
unmasked_text = processor.tokenizer.decode(unmasked_tokens)
print(f"\nUnmasked text (should be assistant response): {repr(unmasked_text)}")

# Check if "Red" is in unmasked text
if "Red" in unmasked_text or "red" in unmasked_text.lower():
    print("✅ SUCCESS: Assistant answer 'Red' is unmasked")
else:
    print("❌ FAILURE: Assistant answer 'Red' is not in unmasked text")

# Check if question is masked
question_tokens = processor.tokenizer.encode("What color is this?", add_special_tokens=False)
question_in_unmasked = any(token_id in unmasked_tokens for token_id in question_tokens)
if not question_in_unmasked:
    print("✅ SUCCESS: User question is masked")
else:
    print("❌ WARNING: Some user question tokens are unmasked")

print("\n" + "=" * 80)
