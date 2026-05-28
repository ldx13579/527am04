"""
Visualize cross-attention heatmaps for noun phrase grounding.
Shows which image patches each noun phrase attends to.

Usage:
    python visualize_grounding.py --checkpoint checkpoints/best_model.pt --num_samples 8
"""
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from transformers import AutoTokenizer

from config import Config
from dataset import load_flickr8k
from model import CLIPModel
from grounding import NounPhraseExtractor


def get_noun_phrase_attention(attn_weights, noun_masks, attention_mask):
    """
    Extract attention maps for noun phrase tokens.
    attn_weights: (1, seq_len, 196)
    noun_masks: (1, seq_len)
    Returns: (196,) average attention over noun phrase tokens
    """
    valid = (noun_masks[0] * attention_mask[0].float()).bool()
    if valid.sum() == 0:
        return attn_weights[0].mean(dim=0).detach().cpu().numpy()
    np_attn = attn_weights[0][valid]  # (num_np_tokens, 196)
    avg_attn = np_attn.mean(dim=0)  # (196,)
    return avg_attn.detach().cpu().numpy()


def get_per_phrase_attention(attn_weights, caption, tokenizer, nlp, max_len=64):
    """
    Get separate attention maps for each noun phrase.
    Returns list of (phrase_text, heatmap_14x14)
    """
    doc = nlp(caption)
    noun_chunks = list(doc.noun_chunks)
    if not noun_chunks:
        return []

    tokens = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = tokens["offset_mapping"].squeeze(0)

    results = []
    for chunk in noun_chunks:
        start_char, end_char = chunk.start_char, chunk.end_char
        chunk_mask = torch.zeros(max_len, dtype=torch.bool)
        for idx in range(max_len):
            tok_start, tok_end = offsets[idx].tolist()
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_start >= start_char and tok_end <= end_char:
                chunk_mask[idx] = True

        if chunk_mask.sum() == 0:
            continue

        phrase_attn = attn_weights[0][chunk_mask]  # (num_tokens, 196)
        avg_attn = phrase_attn.mean(dim=0).detach().cpu().numpy()
        heatmap = avg_attn.reshape(14, 14)
        results.append((chunk.text, heatmap))

    return results


def visualize_samples(model, eval_dataset, tokenizer, config, device, num_samples=8, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    np_extractor = NounPhraseExtractor()
    nlp = np_extractor.nlp

    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    indices = list(range(min(num_samples, len(eval_dataset.images))))

    for sample_idx in indices:
        img_pil = eval_dataset.images[sample_idx]
        caption = eval_dataset.captions[eval_dataset.img2txt[sample_idx][0]]

        img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
        tokens = tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=config.max_text_len, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            _, _, _, attn_weights = model(
                img_tensor, input_ids, attention_mask, return_cross_attn=True
            )

        noun_masks = np_extractor.extract([caption], tokenizer, config.max_text_len).to(device)
        overall_heatmap = get_noun_phrase_attention(attn_weights, noun_masks, attention_mask)
        overall_heatmap = overall_heatmap.reshape(14, 14)

        per_phrase = get_per_phrase_attention(attn_weights, caption, tokenizer, nlp, config.max_text_len)

        num_phrases = len(per_phrase)
        fig_cols = 2 + num_phrases
        fig, axes = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))

        display_img = img_pil.resize((224, 224))
        axes[0].imshow(display_img)
        axes[0].set_title("Original Image", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(display_img)
        hm = axes[1].imshow(
            overall_heatmap, cmap="jet", alpha=0.5,
            extent=[0, 224, 224, 0], interpolation="bilinear"
        )
        axes[1].set_title("All Noun Phrases", fontsize=10)
        axes[1].axis("off")

        for i, (phrase, heatmap) in enumerate(per_phrase):
            ax = axes[2 + i]
            ax.imshow(display_img)
            ax.imshow(
                heatmap, cmap="jet", alpha=0.5,
                extent=[0, 224, 224, 0], interpolation="bilinear"
            )
            ax.set_title(f'"{phrase}"', fontsize=9)
            ax.axis("off")

        caption_short = caption[:80] + "..." if len(caption) > 80 else caption
        fig.suptitle(f"Sample {sample_idx}: {caption_short}", fontsize=11, y=0.02)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"grounding_sample_{sample_idx:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nAll visualizations saved to {output_dir}/")


def visualize_grid(model, eval_dataset, tokenizer, config, device, num_samples=4, output_dir="visualizations"):
    """Create a single grid figure showing multiple samples."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    np_extractor = NounPhraseExtractor()

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_samples = min(num_samples, len(eval_dataset.images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, sample_idx in enumerate(range(num_samples)):
        img_pil = eval_dataset.images[sample_idx]
        caption = eval_dataset.captions[eval_dataset.img2txt[sample_idx][0]]

        img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
        tokens = tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=config.max_text_len, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            _, _, _, attn_weights = model(
                img_tensor, input_ids, attention_mask, return_cross_attn=True
            )

        noun_masks = np_extractor.extract([caption], tokenizer, config.max_text_len).to(device)
        heatmap = get_noun_phrase_attention(attn_weights, noun_masks, attention_mask).reshape(14, 14)

        display_img = img_pil.resize((224, 224))

        axes[row, 0].imshow(display_img)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Image", fontsize=12)

        axes[row, 1].imshow(heatmap, cmap="jet", interpolation="bilinear")
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Attention Heatmap", fontsize=12)

        axes[row, 2].imshow(display_img)
        axes[row, 2].imshow(
            heatmap, cmap="jet", alpha=0.5,
            extent=[0, 224, 224, 0], interpolation="bilinear"
        )
        axes[row, 2].axis("off")
        if row == 0:
            axes[row, 2].set_title("Overlay", fontsize=12)

        caption_short = caption[:60] + "..." if len(caption) > 60 else caption
        axes[row, 0].set_ylabel(f'"{caption_short}"', fontsize=8, rotation=0, labelpad=80, va="center")

    plt.suptitle("Cross-Attention Noun Phrase Grounding Heatmaps", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "grounding_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--grid", action="store_true", help="Also generate a grid overview")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, eval_dataset, tokenizer = load_flickr8k(config)

    print(f"Loading model from {args.checkpoint}...")
    model = CLIPModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    visualize_samples(model, eval_dataset, tokenizer, config, device,
                      num_samples=args.num_samples, output_dir=args.output_dir)

    if args.grid:
        visualize_grid(model, eval_dataset, tokenizer, config, device,
                       num_samples=min(4, args.num_samples), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
