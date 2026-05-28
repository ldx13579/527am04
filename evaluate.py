import torch
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def evaluate(model, eval_dataset, device, config):
    """
    Unified evaluation using encode_image() and encode_text().
    Works identically for CLIPModel and BaselineCLIPModel.
    Computes I2T and T2I Recall@1 and Recall@5.
    """
    model.eval()
    try:
        all_images = eval_dataset.get_image_tensors()
        all_input_ids, all_attention_mask = eval_dataset.get_text_tokens()

        # Encode all images via unified encode_image
        image_embeds = []
        img_loader = DataLoader(TensorDataset(all_images), batch_size=config.batch_size, shuffle=False)
        for (batch_imgs,) in img_loader:
            embeds = model.encode_image(batch_imgs.to(device))
            image_embeds.append(embeds.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)

        # Encode all texts via unified encode_text
        text_embeds = []
        txt_loader = DataLoader(
            TensorDataset(all_input_ids, all_attention_mask),
            batch_size=config.batch_size,
            shuffle=False,
        )
        for batch_ids, batch_mask in txt_loader:
            embeds = model.encode_text(batch_ids.to(device), batch_mask.to(device))
            text_embeds.append(embeds.cpu())
        text_embeds = torch.cat(text_embeds, dim=0)

        sim = (image_embeds @ text_embeds.T) / model.temperature.cpu()

        num_images = len(eval_dataset.images)
        num_texts = len(eval_dataset.captions)

        # Image-to-Text R@1 and R@5
        i2t_r1, i2t_r5 = 0, 0
        for i in range(num_images):
            ranking = sim[i].argsort(descending=True)
            top1 = ranking[0].item()
            top5 = ranking[:5].tolist()
            if top1 in eval_dataset.img2txt[i]:
                i2t_r1 += 1
            if any(t in eval_dataset.img2txt[i] for t in top5):
                i2t_r5 += 1
        i2t_r1 /= num_images
        i2t_r5 /= num_images

        # Text-to-Image R@1 and R@5
        t2i_r1, t2i_r5 = 0, 0
        for j in range(num_texts):
            ranking = sim[:, j].argsort(descending=True)
            top1 = ranking[0].item()
            top5 = ranking[:5].tolist()
            if top1 == eval_dataset.txt2img[j]:
                t2i_r1 += 1
            if eval_dataset.txt2img[j] in top5:
                t2i_r5 += 1
        t2i_r1 /= num_texts
        t2i_r5 /= num_texts

        return {
            "i2t_r1": i2t_r1, "i2t_r5": i2t_r5,
            "t2i_r1": t2i_r1, "t2i_r5": t2i_r5,
        }
    finally:
        model.train()


@torch.no_grad()
def evaluate_paired(model, eval_dataset, device, config):
    """
    Paired evaluation using encode_pair() — both image and text go through
    cross-attention fusion before normalization.
    Slower (O(N*M) pairs) but utilizes the full bidirectional attention.
    Only practical for small eval sets; uses re-ranking over top-K candidates.
    """
    model.eval()
    try:
        all_images = eval_dataset.get_image_tensors()
        all_input_ids, all_attention_mask = eval_dataset.get_text_tokens()

        num_images = len(eval_dataset.images)
        num_texts = len(eval_dataset.captions)

        # First-pass: independent encoding for candidate retrieval
        image_embeds = []
        img_loader = DataLoader(TensorDataset(all_images), batch_size=config.batch_size, shuffle=False)
        for (batch_imgs,) in img_loader:
            embeds = model.encode_image(batch_imgs.to(device))
            image_embeds.append(embeds.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)

        text_embeds = []
        txt_loader = DataLoader(
            TensorDataset(all_input_ids, all_attention_mask),
            batch_size=config.batch_size, shuffle=False,
        )
        for batch_ids, batch_mask in txt_loader:
            embeds = model.encode_text(batch_ids.to(device), batch_mask.to(device))
            text_embeds.append(embeds.cpu())
        text_embeds = torch.cat(text_embeds, dim=0)

        coarse_sim = image_embeds @ text_embeds.T

        # Re-rank top-K candidates using paired cross-attention encoding
        top_k = min(20, num_texts)

        # I2T re-ranking
        i2t_r1, i2t_r5 = 0, 0
        for i in range(num_images):
            candidates = coarse_sim[i].argsort(descending=True)[:top_k]
            img_batch = all_images[i].unsqueeze(0).expand(top_k, -1, -1, -1).to(device)
            txt_ids_batch = all_input_ids[candidates].to(device)
            txt_mask_batch = all_attention_mask[candidates].to(device)

            img_emb, txt_emb = model.encode_pair(img_batch, txt_ids_batch, txt_mask_batch)
            paired_sim = (img_emb * txt_emb).sum(dim=-1)
            reranked = candidates[paired_sim.argsort(descending=True)]

            top1 = reranked[0].item()
            top5 = reranked[:5].tolist()
            if top1 in eval_dataset.img2txt[i]:
                i2t_r1 += 1
            if any(t in eval_dataset.img2txt[i] for t in top5):
                i2t_r5 += 1
        i2t_r1 /= num_images
        i2t_r5 /= num_images

        # T2I re-ranking
        t2i_r1, t2i_r5 = 0, 0
        for j in range(num_texts):
            candidates = coarse_sim[:, j].argsort(descending=True)[:top_k]
            img_batch = all_images[candidates].to(device)
            txt_ids_batch = all_input_ids[j].unsqueeze(0).expand(top_k, -1).to(device)
            txt_mask_batch = all_attention_mask[j].unsqueeze(0).expand(top_k, -1).to(device)

            img_emb, txt_emb = model.encode_pair(img_batch, txt_ids_batch, txt_mask_batch)
            paired_sim = (img_emb * txt_emb).sum(dim=-1)
            reranked = candidates[paired_sim.argsort(descending=True)]

            top1 = reranked[0].item()
            top5 = reranked[:5].tolist()
            if top1 == eval_dataset.txt2img[j]:
                t2i_r1 += 1
            if eval_dataset.txt2img[j] in top5:
                t2i_r5 += 1
        t2i_r1 /= num_texts
        t2i_r5 /= num_texts

        return {
            "i2t_r1": i2t_r1, "i2t_r5": i2t_r5,
            "t2i_r1": t2i_r1, "t2i_r5": t2i_r5,
        }
    finally:
        model.train()
