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
            embeds = model.image_encoder(batch_imgs.to(device))
            image_embeds.append(embeds.cpu())
        image_embeds = torch.cat(image_embeds, dim=0)

        text_embeds = []
        txt_loader = DataLoader(
            TensorDataset(all_input_ids, all_attention_mask),
            batch_size=config.batch_size,
            shuffle=False,
        )
        for batch_ids, batch_mask in txt_loader:
            embeds = model.text_encoder(batch_ids.to(device), batch_mask.to(device))
            text_embeds.append(embeds.cpu())
        text_embeds = torch.cat(text_embeds, dim=0)

        sim = (image_embeds @ text_embeds.T) / model.temperature.cpu()

        # Image-to-Text R@1
        i2t_correct = 0
        num_images = len(eval_dataset.images)
        for i in range(num_images):
            top1 = sim[i].argmax().item()
            if top1 in eval_dataset.img2txt[i]:
                i2t_correct += 1
        i2t_r1 = i2t_correct / num_images

        # Text-to-Image R@1
        t2i_correct = 0
        num_texts = len(eval_dataset.captions)
        for j in range(num_texts):
            top1 = sim[:, j].argmax().item()
            if top1 == eval_dataset.txt2img[j]:
                t2i_correct += 1
        t2i_r1 = t2i_correct / num_texts

        return {"i2t_r1": i2t_r1, "t2i_r1": t2i_r1}
    finally:
        model.train()
