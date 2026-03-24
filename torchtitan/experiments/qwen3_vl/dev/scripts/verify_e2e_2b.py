"""
End-to-end logits comparison: HF vs TorchTitan using Qwen3-VL-2B (dense, non-MOE).

Loads both models with the same weights, runs the same image+prompt input,
compares logits token-by-token.
"""

import torch
import sys
from PIL import Image


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    torch.manual_seed(42)
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    print("=" * 60)
    print("End-to-End Logits Comparison: Qwen3-VL-2B")
    print("=" * 60)

    # =========================================================================
    # 1. Load HF model
    # =========================================================================
    print("\n=== 1. Loading HF model ===")
    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration

    processor = Qwen3VLProcessor.from_pretrained(model_name, trust_remote_code=True)
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )
    hf_model.eval()
    print(f"  HF model loaded. Memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GiB")

    # =========================================================================
    # 2. Create test input
    # =========================================================================
    print("\n=== 2. Creating test input ===")
    img = Image.new("RGB", (224, 224), color=(200, 50, 100))

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in one word."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

    print(f"  input_ids: {inputs.input_ids.shape}")
    print(f"  pixel_values: {inputs.pixel_values.shape}")
    print(f"  image_grid_thw: {inputs.image_grid_thw}")

    # =========================================================================
    # 3. Run HF forward
    # =========================================================================
    print("\n=== 3. HF Forward Pass ===")
    with torch.no_grad():
        hf_out = hf_model(**inputs)
    hf_logits = hf_out.logits
    print(f"  logits: {hf_logits.shape}")

    # Top-5 last token
    last_logits_hf = hf_logits[0, -1]
    top5_vals, top5_ids = last_logits_hf.topk(5)
    print("  HF Top-5 (last token):")
    for i, (v, idx) in enumerate(zip(top5_vals, top5_ids)):
        tok = processor.tokenizer.decode([idx.item()])
        print(f"    {i+1}. '{tok}' (id={idx.item()}, logit={v.item():.2f})")

    # =========================================================================
    # 4. Build TorchTitan model with 2B config
    # =========================================================================
    print("\n=== 4. Building TorchTitan model ===")
    from torchtitan.experiments.qwen3_vl.model.args import Qwen3VLModelArgs, Qwen3VLVisionArgs
    from torchtitan.experiments.qwen3_vl.model.model import Qwen3VLModel

    tt_args = Qwen3VLModelArgs(
        # Text model (2B dense)
        vocab_size=151936,
        max_seq_len=262144,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=5000000,
        moe_enabled=False,
        enable_weight_tying=True,
        # Vision config (2B)
        vision_config=Qwen3VLVisionArgs(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indexes=[5, 11, 17],
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        deepstack_visual_indexes=[5, 11, 17],
    )

    tt_model = Qwen3VLModel(tt_args).to(device, dtype).eval()
    print(f"  TT model created. Params: {sum(p.numel() for p in tt_model.parameters()):,}")

    # =========================================================================
    # 5. Copy weights from HF → TorchTitan
    # =========================================================================
    print("\n=== 5. Copying weights HF → TorchTitan ===")
    from torchtitan.experiments.qwen3_vl.model.state_dict_adapter import Qwen3VLStateDictAdapter

    adapter = Qwen3VLStateDictAdapter(tt_args, model_name)
    hf_sd = hf_model.state_dict()
    tt_sd = adapter.from_hf(hf_sd)

    # Load into TorchTitan model
    missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
    print(f"  Missing keys: {len(missing)}")
    if missing:
        for k in missing[:5]:
            print(f"    {k}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"    {k}")

    print(f"  Memory after loading: {torch.cuda.memory_allocated(device)/1e9:.1f} GiB")

    # =========================================================================
    # 6. Run TorchTitan forward
    # =========================================================================
    print("\n=== 6. TorchTitan Forward Pass ===")

    # Prepare inputs matching TorchTitan's forward signature
    from torchtitan.experiments.qwen3_vl.model.model import get_rope_index

    # Compute position_ids
    position_ids, _ = get_rope_index(
        inputs.input_ids,
        inputs.image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )

    with torch.no_grad():
        tt_logits = tt_model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values.to(dtype),
            image_grid_thw=inputs.image_grid_thw,
            position_ids=position_ids,
        )
    print(f"  logits: {tt_logits.shape}")

    # Top-5 last token
    last_logits_tt = tt_logits[0, -1]
    top5_vals_tt, top5_ids_tt = last_logits_tt.topk(5)
    print("  TT Top-5 (last token):")
    for i, (v, idx) in enumerate(zip(top5_vals_tt, top5_ids_tt)):
        tok = processor.tokenizer.decode([idx.item()])
        print(f"    {i+1}. '{tok}' (id={idx.item()}, logit={v.item():.2f})")

    # =========================================================================
    # 7. Compare logits
    # =========================================================================
    print("\n=== 7. Logits Comparison ===")
    hf_l = hf_logits.float()
    tt_l = tt_logits.float()

    if hf_l.shape != tt_l.shape:
        print(f"  Shape mismatch: HF={hf_l.shape} TT={tt_l.shape}")
        # Truncate to min length
        min_len = min(hf_l.shape[1], tt_l.shape[1])
        hf_l = hf_l[:, :min_len]
        tt_l = tt_l[:, :min_len]

    # Overall comparison
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_l.flatten().unsqueeze(0), tt_l.flatten().unsqueeze(0)
    ).item()
    abs_diff = (hf_l - tt_l).abs()
    print(f"  Overall cosine_similarity: {cos_sim:.8f}")
    print(f"  max_abs_diff: {abs_diff.max().item():.4f}")
    print(f"  mean_abs_diff: {abs_diff.mean().item():.4f}")

    # Per-position comparison
    per_pos_cos = torch.nn.functional.cosine_similarity(
        hf_l[0], tt_l[0], dim=-1
    )
    print(f"  Per-position cosine_sim: min={per_pos_cos.min():.6f}, mean={per_pos_cos.mean():.6f}")

    # Top-1 agreement
    hf_top1 = hf_l[0].argmax(dim=-1)
    tt_top1 = tt_l[0].argmax(dim=-1)
    agreement = (hf_top1 == tt_top1).float().mean().item()
    print(f"  Top-1 token agreement: {agreement:.2%}")

    # Last token agreement
    last_agree = hf_top1[-1].item() == tt_top1[-1].item()
    print(f"  Last token top-1 match: {last_agree} (HF={hf_top1[-1].item()}, TT={tt_top1[-1].item()})")

    # =========================================================================
    # 8. Verdict
    # =========================================================================
    print("\n" + "=" * 60)
    passed = cos_sim > 0.99 and agreement > 0.9
    if passed:
        print(f"PASS: Logits match (cos_sim={cos_sim:.6f}, top-1 agreement={agreement:.1%})")
    else:
        print(f"FAIL: Logits diverge (cos_sim={cos_sim:.6f}, top-1 agreement={agreement:.1%})")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
