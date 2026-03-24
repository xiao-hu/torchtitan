"""
End-to-end generation test: compare HF Qwen3VL vs TorchTitan native.

Loads the full HF model, runs a single forward pass on the same image+prompt,
and compares the logits. Uses HF model as reference since it's proven.

Runs on a single GPU with the vision encoder only (language model too large).
Instead, we compare the full Qwen3VLModel forward (vision + embedding injection
+ first few language model layers) by loading a small subset of weights.

Actually: the simplest meaningful test is to run the HF pipeline end-to-end
and compare the vision encoder output + embedded input to the language model.
"""

import torch
import sys


def main():
    device = "cuda:0"
    dtype = torch.bfloat16  # Match training dtype
    torch.manual_seed(42)

    print("=" * 60)
    print("End-to-End Generation Verification")
    print("=" * 60)

    # =========================================================================
    # 1. Load HF model for reference
    # =========================================================================
    print("\n=== Loading HF Qwen3VL ===")
    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image
    import requests

    checkpoint = "/checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct"

    processor = Qwen3VLProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    # Load HF model (will use a lot of memory but we only need one forward pass)
    print("  Loading HF model (this takes a while)...")
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map={"": device},  # Load everything on one GPU
        trust_remote_code=True,
    )
    hf_model.eval()
    print(f"  HF model loaded. Memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GiB")

    # =========================================================================
    # 2. Create test input with a real image
    # =========================================================================
    print("\n=== Creating test input ===")
    # Create a simple test image (solid color with a shape)
    img = Image.new("RGB", (224, 224), color=(128, 64, 192))

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What color is this image?"},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

    print(f"  input_ids: {inputs.input_ids.shape}")
    print(f"  pixel_values: {inputs.pixel_values.shape}")
    print(f"  image_grid_thw: {inputs.image_grid_thw}")

    # =========================================================================
    # 3. Run HF model forward (just 1 step, get logits)
    # =========================================================================
    print("\n=== HF Forward Pass ===")
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        hf_logits = hf_outputs.logits
    print(f"  logits: {hf_logits.shape}")

    # Get top-5 predictions for the last token
    last_logits = hf_logits[0, -1]
    top5_values, top5_indices = last_logits.topk(5)
    print("  Top-5 predictions (last token):")
    for i, (val, idx) in enumerate(zip(top5_values, top5_indices)):
        token = processor.tokenizer.decode([idx.item()])
        print(f"    {i+1}. '{token}' (id={idx.item()}, logit={val.item():.2f})")

    # =========================================================================
    # 4. Extract HF vision encoder output for comparison
    # =========================================================================
    print("\n=== Comparing Vision Encoder Outputs ===")
    # Run HF vision encoder separately
    hf_vision_output = hf_model.model.visual(
        inputs.pixel_values.to(dtype), inputs.image_grid_thw
    )
    hf_merged = hf_vision_output.pooler_output
    hf_deepstack = hf_vision_output.deepstack_features

    # Run native vision encoder with same weights
    from torchtitan.experiments.qwen3_vl.model.native_vision import Qwen3VLNativeVisionEncoder
    from torchtitan.experiments.qwen3_vl.model.args import Qwen3VLVisionArgs

    native_args = Qwen3VLVisionArgs(
        depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
        in_channels=3, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=2048, num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
    )
    native_vision = Qwen3VLNativeVisionEncoder(native_args).to(device, dtype).eval()

    # Copy weights from HF
    hf_sd = hf_model.model.visual.state_dict()
    mapped = {}
    for hf_key, hf_val in hf_sd.items():
        native_key = hf_key.replace(".linear_fc1.", ".fc1.").replace(".linear_fc2.", ".fc2.")
        if native_key == "patch_embed.proj.weight":
            hf_val = hf_val.reshape(hf_val.shape[0], -1)
            native_key = "patch_embed.weight"
        elif native_key == "patch_embed.proj.bias":
            native_key = "patch_embed.bias"
        if native_key in native_vision.state_dict():
            mapped[native_key] = hf_val
    native_vision.load_state_dict(mapped, strict=False)

    with torch.no_grad():
        native_merged, native_deepstack = native_vision(
            inputs.pixel_values.to(dtype), inputs.image_grid_thw
        )

    # Compare
    def compare(name, a, b):
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
        ).item()
        diff = (a.float() - b.float()).abs()
        print(f"  {name}: cos_sim={cos:.8f}, max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
        return cos > 0.99

    all_pass = True
    all_pass &= compare("merged", native_merged, hf_merged)
    for i, (nd, hd) in enumerate(zip(native_deepstack, hf_deepstack)):
        all_pass &= compare(f"deepstack[{i}]", nd, hd)

    # =========================================================================
    # 5. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("PASS: Native vision encoder matches HF on real image input")
        print(f"HF model generates: '{processor.tokenizer.decode([top5_indices[0].item()])}'")
    else:
        print("FAIL: Native vision encoder diverges from HF")
    print("=" * 60)

    # Cleanup
    del hf_model
    torch.cuda.empty_cache()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
