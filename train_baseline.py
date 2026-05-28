"""
Train baseline model (without cross-attention/grounding) for comparison.
Uses the same unified encode_image/encode_text interface.
Saves checkpoint to checkpoints/baseline.pt

Usage:
    python train_baseline.py
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import Config
from dataset import load_flickr8k, get_train_loader
from model import BaselineCLIPModel
from loss import CLIPLossWithHardNegatives
from evaluate import evaluate


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_baseline():
    config = Config()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] Using device: {device}")

    train_dataset, eval_dataset, tokenizer = load_flickr8k(config)
    train_iter = get_train_loader(train_dataset, config)

    model = BaselineCLIPModel(config).to(device)
    criterion = CLIPLossWithHardNegatives(config.hard_negative_weight)

    param_groups = [
        {"params": model.image_encoder.parameters(), "lr": config.learning_rate},
        {"params": model.text_encoder.parameters(), "lr": config.learning_rate * 0.1},
        {"params": [model.log_temperature], "lr": config.learning_rate},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    scaler = GradScaler(enabled=config.use_amp)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_recall = 0.0

    pbar = tqdm(range(1, config.total_steps + 1), desc="Baseline Training")
    for step in pbar:
        model.train()
        images, input_ids, attention_mask, _ = next(train_iter)
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            image_embeds, text_embeds, temperature = model(images, input_ids, attention_mask)
            loss, base_loss, hard_loss = criterion(image_embeds, text_embeds, temperature)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                base=f"{base_loss:.3f}",
                hard=f"{hard_loss:.3f}",
                temp=f"{temperature.item():.2f}",
            )

        if step % config.eval_every == 0 or step == config.total_steps:
            metrics = evaluate(model, eval_dataset, device, config)
            avg_r1 = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2
            avg_r5 = (metrics["i2t_r5"] + metrics["t2i_r5"]) / 2
            print(
                f"\n[Baseline Step {step}] "
                f"I2T R@1: {metrics['i2t_r1']*100:.1f}% R@5: {metrics['i2t_r5']*100:.1f}% | "
                f"T2I R@1: {metrics['t2i_r1']*100:.1f}% R@5: {metrics['t2i_r5']*100:.1f}% | "
                f"Avg R@5: {avg_r5*100:.1f}%"
            )
            if avg_r1 > best_recall:
                best_recall = avg_r1
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "best_recall": best_recall,
                    },
                    os.path.join(config.checkpoint_dir, "baseline.pt"),
                )
                print(f"  -> New best baseline! Saved.")

    print(f"\nBaseline training complete. Best Avg R@1: {best_recall*100:.1f}%")


if __name__ == "__main__":
    train_baseline()
