import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLossWithHardNegatives(nn.Module):
    def __init__(self, hard_negative_weight=0.5):
        super().__init__()
        self.hard_negative_weight = hard_negative_weight

    def forward(self, image_embeds, text_embeds, temperature):
        logits = (image_embeds @ text_embeds.T) / temperature
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        base_loss = (loss_i2t + loss_t2i) / 2

        with torch.no_grad():
            mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
            logits_masked = logits.detach().clone()
            logits_masked[~mask] = -1e9
            hard_neg_t_idx = logits_masked.argmax(dim=1)
            hard_neg_i_idx = logits_masked.T.argmax(dim=1)

        arange = torch.arange(B, device=logits.device)
        pos_i2t = logits[arange, labels]
        neg_i2t = logits[arange, hard_neg_t_idx]
        hard_loss_i2t = -torch.log(
            torch.exp(pos_i2t) / (torch.exp(pos_i2t) + torch.exp(neg_i2t))
        ).mean()

        pos_t2i = logits[labels, arange]
        neg_t2i = logits[hard_neg_i_idx, arange]
        hard_loss_t2i = -torch.log(
            torch.exp(pos_t2i) / (torch.exp(pos_t2i) + torch.exp(neg_t2i))
        ).mean()

        hard_loss = (hard_loss_i2t + hard_loss_t2i) / 2
        total_loss = base_loss + self.hard_negative_weight * hard_loss

        return total_loss, base_loss.item(), hard_loss.item()
