"""
Evaluate retrieval accuracy specifically on rare/uncommon objects.
Tests whether KG + modality dropout improves retrieval for objects like zebra,
giraffe, penguin, etc. that appear infrequently in typical datasets.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from knowledge_graph import NODE_TO_IDX, extract_kg_nodes_from_caption

# Rare objects that appear infrequently in Flickr8k but exist in our KG
RARE_OBJECTS = [
    "zebra", "giraffe", "penguin", "eagle", "owl", "parrot", "snake",
    "frog", "turtle", "whale", "dolphin", "shark", "crab", "butterfly",
    "helicopter", "canoe", "yacht", "tractor", "skateboard", "surfboard",
    "watermelon", "strawberry", "hamburger", "castle", "tower",
]


def find_rare_object_samples(eval_dataset):
    """Find captions in the eval set that mention rare objects.
    Returns dict: object_name -> list of caption indices."""
    rare_samples = {}
    for obj in RARE_OBJECTS:
        if obj not in NODE_TO_IDX:
            continue
        matching_captions = []
        for idx, caption in enumerate(eval_dataset.captions):
            if obj.lower() in caption.lower():
                matching_captions.append(idx)
        if matching_captions:
            rare_samples[obj] = matching_captions
    return rare_samples


@torch.no_grad()
def evaluate_rare_objects(model, eval_dataset, device, config):
    """
    Evaluate retrieval performance specifically on rare object queries.
    Reports per-object and aggregate Recall@1, Recall@5 for T2I retrieval.
    For KGEnhancedCLIPModel, passes KG nodes extracted from captions.
    """
    model.eval()
    has_kg = hasattr(model, 'kg_gcn')
    kg_max_nodes = getattr(config, 'kg_max_nodes', 5)

    try:
        rare_samples = find_rare_object_samples(eval_dataset)
        if not rare_samples:
            print("No rare objects found in evaluation set.")
            return {}

        all_images = eval_dataset.get_image_tensors()
        all_input_ids, all_attention_mask = eval_dataset.get_text_tokens()

        # Extract KG nodes for all captions
        kg_node_indices_all = None
        kg_node_mask_all = None
        if has_kg:
            kg_node_indices_all, kg_node_mask_all = extract_kg_nodes_batch(
                eval_dataset.captions, max_nodes=kg_max_nodes
            )

        # Encode all images (no KG at image side for independent retrieval)
        image_embeds = []
        img_loader = DataLoader(
            TensorDataset(all_images), batch_size=config.batch_size, shuffle=False
        )
        for (batch_imgs,) in img_loader:
            embeds = model.encode_image(batch_imgs.to(device))
            image_embeds.append(embeds.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)

        # Encode all texts with KG nodes
        text_embeds = []
        if has_kg:
            txt_dataset = TensorDataset(
                all_input_ids, all_attention_mask,
                kg_node_indices_all, kg_node_mask_all
            )
        else:
            txt_dataset = TensorDataset(all_input_ids, all_attention_mask)
        txt_loader = DataLoader(txt_dataset, batch_size=config.batch_size, shuffle=False)

        for batch in txt_loader:
            if has_kg:
                batch_ids, batch_mask, batch_kg_idx, batch_kg_mask = batch
                embeds = model.encode_text(
                    batch_ids.to(device), batch_mask.to(device),
                    kg_node_indices=batch_kg_idx.to(device),
                    kg_node_mask=batch_kg_mask.to(device),
                )
            else:
                batch_ids, batch_mask = batch
                embeds = model.encode_text(batch_ids.to(device), batch_mask.to(device))
            text_embeds.append(embeds.cpu())
        text_embeds = torch.cat(text_embeds, dim=0)

        sim = image_embeds @ text_embeds.T

        # Evaluate T2I for each rare object
        results = {}
        total_r1, total_r5, total_count = 0, 0, 0

        print("\n" + "=" * 60)
        print("Rare Object Retrieval Evaluation (Text-to-Image)")
        print("=" * 60)
        print(f"{'Object':<15} {'Count':<7} {'R@1':<10} {'R@5':<10}")
        print("-" * 42)

        for obj, caption_indices in sorted(rare_samples.items()):
            obj_r1, obj_r5 = 0, 0
            for j in caption_indices:
                ranking = sim[:, j].argsort(descending=True)
                top1 = ranking[0].item()
                top5 = ranking[:5].tolist()
                gt_img = eval_dataset.txt2img[j]
                if top1 == gt_img:
                    obj_r1 += 1
                if gt_img in top5:
                    obj_r5 += 1

            n = len(caption_indices)
            r1_pct = obj_r1 / n * 100
            r5_pct = obj_r5 / n * 100
            results[obj] = {"r1": r1_pct, "r5": r5_pct, "count": n}
            total_r1 += obj_r1
            total_r5 += obj_r5
            total_count += n
            print(f"{obj:<15} {n:<7} {r1_pct:<10.1f} {r5_pct:<10.1f}")

        if total_count > 0:
            avg_r1 = total_r1 / total_count * 100
            avg_r5 = total_r5 / total_count * 100
            print("-" * 42)
            print(f"{'AVERAGE':<15} {total_count:<7} {avg_r1:<10.1f} {avg_r5:<10.1f}")
            results["_average"] = {"r1": avg_r1, "r5": avg_r5, "count": total_count}
        print("=" * 60)

        return results
    finally:
        model.train()


@torch.no_grad()
def compare_rare_objects(model_kg, model_baseline, eval_dataset, device, config):
    """Compare KG-enhanced model vs baseline on rare object retrieval."""
    print("\n>>> KG-Enhanced Model (with Knowledge Graph + Modality Dropout):")
    results_kg = evaluate_rare_objects(model_kg, eval_dataset, device, config)

    print("\n>>> Baseline Model (no KG, no modality dropout):")
    results_baseline = evaluate_rare_objects(model_baseline, eval_dataset, device, config)

    if results_kg and results_baseline:
        print("\n" + "=" * 60)
        print("IMPROVEMENT (KG vs Baseline)")
        print("=" * 60)
        print(f"{'Object':<15} {'Baseline R@5':<14} {'KG R@5':<10} {'Delta':<10}")
        print("-" * 49)
        for obj in sorted(results_kg.keys()):
            if obj.startswith("_"):
                continue
            if obj in results_baseline:
                b_r5 = results_baseline[obj]["r5"]
                k_r5 = results_kg[obj]["r5"]
                delta = k_r5 - b_r5
                sign = "+" if delta >= 0 else ""
                print(f"{obj:<15} {b_r5:<14.1f} {k_r5:<10.1f} {sign}{delta:<10.1f}")
        if "_average" in results_kg and "_average" in results_baseline:
            b_avg = results_baseline["_average"]["r5"]
            k_avg = results_kg["_average"]["r5"]
            delta = k_avg - b_avg
            sign = "+" if delta >= 0 else ""
            print("-" * 49)
            print(f"{'AVERAGE':<15} {b_avg:<14.1f} {k_avg:<10.1f} {sign}{delta:<10.1f}")
        print("=" * 60)


if __name__ == "__main__":
    import os
    from config import Config
    from model import KGEnhancedCLIPModel, BaselineCLIPModel
    from dataset import load_flickr8k

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, eval_dataset, _ = load_flickr8k(config)

    # Load KG-enhanced model
    kg_model = KGEnhancedCLIPModel(config).to(device)
    kg_ckpt_path = os.path.join(config.checkpoint_dir, "best_kg_model.pt")
    if os.path.exists(kg_ckpt_path):
        ckpt = torch.load(kg_ckpt_path, map_location=device)
        kg_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded KG model from {kg_ckpt_path}")

    # Load baseline model
    baseline_model = BaselineCLIPModel(config).to(device)
    baseline_ckpt_path = os.path.join(config.checkpoint_dir, "baseline.pt")
    if os.path.exists(baseline_ckpt_path):
        ckpt = torch.load(baseline_ckpt_path, map_location=device)
        baseline_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded baseline from {baseline_ckpt_path}")

    compare_rare_objects(kg_model, baseline_model, eval_dataset, device, config)
