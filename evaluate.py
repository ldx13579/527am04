import torch
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def evaluate(model, eval_dataset, device, config):
    model.eval()
    try:
        all_images = eval_dataset.get_image_tensors()
        all_input_ids, all_attention_mask = eval_dataset.get_text_tokens()

        image_embeds = []
        img_loader = DataLoader(TensorDataset(all_images), batch_size=config.batch_size, shuffle=False)
        for (batch_imgs,) in img_loader:
            batch_imgs = batch_imgs.to(device)
            embeds = model.image_encoder(batch_imgs, return_patches=False)
            image_embeds.append(embeds.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)

        text_embeds = []
        txt_loader = DataLoader(
            TensorDataset(all_input_ids, all_attention_mask),
            batch_size=config.batch_size,
            shuffle=False,
        )
        for batch_ids, batch_mask in txt_loader:
            embeds = model.text_encoder(batch_ids.to(device), batch_mask.to(device), return_tokens=False)
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
