"""
Compare baseline CLIP vs. enhanced model (bidirectional cross-attention + grounding).
Both models use the same unified encode_image/encode_text interface for fair comparison.

Usage:
    python compare.py --baseline checkpoints/baseline.pt --enhanced checkpoints/best_model.pt
"""
import argparse
import torch

from config import Config
from dataset import load_flickr8k
from model import CLIPModel, BaselineCLIPModel
from evaluate import evaluate


def load_enhanced(checkpoint_path, config, device):
    model = CLIPModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_baseline(checkpoint_path, config, device):
    model = BaselineCLIPModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="checkpoints/baseline.pt")
    parser.add_argument("--enhanced", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_dataset, _ = load_flickr8k(config)

    print("=" * 65)
    print("Model Comparison: Baseline vs. Bidirectional CrossAttn + Grounding")
    print("=" * 65)

    results = {}

    print(f"\nLoading Baseline from {args.baseline}...")
    baseline_model = load_baseline(args.baseline, config, device)
    metrics = evaluate(baseline_model, eval_dataset, device, config)
    results["Baseline"] = metrics
    print(f"  I2T R@1: {metrics['i2t_r1']*100:.1f}%  R@5: {metrics['i2t_r5']*100:.1f}%")
    print(f"  T2I R@1: {metrics['t2i_r1']*100:.1f}%  R@5: {metrics['t2i_r5']*100:.1f}%")
    avg_r5 = (metrics['i2t_r5'] + metrics['t2i_r5']) / 2
    print(f"  Avg R@5: {avg_r5*100:.1f}%")

    print(f"\nLoading Enhanced from {args.enhanced}...")
    enhanced_model = load_enhanced(args.enhanced, config, device)
    metrics = evaluate(enhanced_model, eval_dataset, device, config)
    results["Enhanced"] = metrics
    print(f"  I2T R@1: {metrics['i2t_r1']*100:.1f}%  R@5: {metrics['i2t_r5']*100:.1f}%")
    print(f"  T2I R@1: {metrics['t2i_r1']*100:.1f}%  R@5: {metrics['t2i_r5']*100:.1f}%")
    avg_r5 = (metrics['i2t_r5'] + metrics['t2i_r5']) / 2
    print(f"  Avg R@5: {avg_r5*100:.1f}%")

    base = results["Baseline"]
    enhanced = results["Enhanced"]
    print("\n" + "=" * 65)
    print("Improvement (Enhanced - Baseline):")
    print("-" * 65)
    for key in ["i2t_r1", "i2t_r5", "t2i_r1", "t2i_r5"]:
        delta = (enhanced[key] - base[key]) * 100
        print(f"  {key}: {delta:+.1f}%")
    avg_r5_base = (base['i2t_r5'] + base['t2i_r5']) / 2
    avg_r5_enhanced = (enhanced['i2t_r5'] + enhanced['t2i_r5']) / 2
    print(f"  Avg R@5: {(avg_r5_enhanced - avg_r5_base)*100:+.1f}% (target: +5%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
