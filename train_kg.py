import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import Config
from dataset import load_flickr8k, get_train_loader
from model import KGEnhancedCLIPModel
from loss import CLIPLossWithHardNegatives
from grounding import NounPhraseExtractor, GroundingLoss, SaliencyEstimator, RegionSupervisionLoss
from knowledge_graph import extract_kg_nodes_batch
from evaluate import evaluate


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    config = Config()
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"KG dim: {config.kg_dim}, Modality drop prob: {config.modality_drop_prob}")

    train_dataset, eval_dataset, tokenizer = load_flickr8k(config)
    train_iter = get_train_loader(train_dataset, config)

    model = KGEnhancedCLIPModel(config).to(device)
    criterion = CLIPLossWithHardNegatives(config.hard_negative_weight)
    grounding_loss_fn = GroundingLoss()
    region_loss_fn = RegionSupervisionLoss()
    saliency_estimator = SaliencyEstimator(grid_size=14).to(device)
    np_extractor = NounPhraseExtractor()

    param_groups = [
        {"params": model.image_encoder.parameters(), "lr": config.learning_rate},
        {"params": model.text_encoder.parameters(), "lr": config.learning_rate * 0.1},
        {"params": model.cross_attention.parameters(), "lr": config.learning_rate},
        {"params": model.kg_gcn.parameters(), "lr": config.learning_rate},
        {"params": model.kg_aggregator.parameters(), "lr": config.learning_rate},
        {"params": model.kg_alignment.parameters(), "lr": config.learning_rate},
        {"params": model.img_kg_classifier.parameters(), "lr": config.learning_rate},
        {"params": model.img_kg_fusion.parameters(), "lr": config.learning_rate},
        {"params": model.txt_kg_fusion.parameters(), "lr": config.learning_rate},
        {"params": model.modality_dropout.parameters(), "lr": config.learning_rate},
        {"params": [model.img_gate, model.txt_gate, model.log_temperature], "lr": config.learning_rate},
        {"params": saliency_estimator.parameters(), "lr": config.learning_rate},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)
    scaler = GradScaler(enabled=config.use_amp)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_recall = 0.0

    pbar = tqdm(range(1, config.total_steps + 1), desc="Training (KG+ModalDrop)")
    for step in pbar:
        model.train()
        saliency_estimator.train()
        images, input_ids, attention_mask, indices = next(train_iter)
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        captions = [train_dataset.captions[i] for i in indices.tolist()]
        noun_masks = np_extractor.extract(captions, tokenizer, config.max_text_len).to(device)

        # Extract KG node indices from captions
        kg_node_indices, kg_node_mask = extract_kg_nodes_batch(
            captions, max_nodes=config.kg_max_nodes
        )
        kg_node_indices = kg_node_indices.to(device)
        kg_node_mask = kg_node_mask.to(device)

        with torch.amp.autocast("cuda", enabled=config.use_amp):
            image_embeds, text_embeds, temperature, t2i_attn, i2t_attn, recon_loss, kg_align_loss, img_kg_cls_loss = model(
                images, input_ids, attention_mask,
                kg_node_indices=kg_node_indices,
                kg_node_mask=kg_node_mask,
                return_cross_attn=True,
            )
            clip_loss, base_loss, hard_loss = criterion(image_embeds, text_embeds, temperature)

            g_loss = grounding_loss_fn(t2i_attn, noun_masks, attention_mask)

            saliency_map = saliency_estimator(images)
            r_loss = region_loss_fn(t2i_attn, saliency_map, noun_masks, attention_mask)

            # Progressive KG alignment weight: ramp from kg_align_weight_min to kg_align_weight_max
            progress = min(step / config.total_steps, 1.0)
            kg_align_w = config.kg_align_weight_min + (config.kg_align_weight_max - config.kg_align_weight_min) * progress

            loss = (clip_loss
                    + config.grounding_weight * g_loss
                    + config.region_weight * r_loss
                    + config.recon_weight * recon_loss
                    + kg_align_w * kg_align_loss
                    + config.img_kg_cls_weight * img_kg_cls_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(saliency_estimator.parameters()),
            config.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                clip=f"{clip_loss.item():.3f}",
                recon=f"{recon_loss.item():.3f}",
                kg_a=f"{kg_align_loss.item():.3f}",
                kg_c=f"{img_kg_cls_loss.item():.3f}",
                kw=f"{kg_align_w:.2f}",
            )

        if step % config.eval_every == 0 or step == config.total_steps:
            metrics = evaluate(model, eval_dataset, device, config)
            avg_r1 = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2
            avg_r5 = (metrics["i2t_r5"] + metrics["t2i_r5"]) / 2
            print(
                f"\n[Step {step}] "
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
                        "saliency_state_dict": saliency_estimator.state_dict(),
                        "best_recall": best_recall,
                        "config": config,
                    },
                    os.path.join(config.checkpoint_dir, "best_kg_model.pt"),
                )
                print(f"  -> New best! Saved checkpoint.")

    print(f"\nTraining complete. Best Avg R@1: {best_recall*100:.1f}%")


if __name__ == "__main__":
    train()
