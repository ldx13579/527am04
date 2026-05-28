import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy


class NounPhraseExtractor:
    """Extract noun phrases from captions using spaCy."""
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, captions, tokenizer, max_len=64):
        """
        For each caption, find noun phrase token positions in the tokenized sequence.
        Returns:
            noun_masks: (B, max_len) binary mask where 1 = token belongs to a noun phrase
        """
        batch_masks = []
        for caption in captions:
            doc = self.nlp(caption)
            noun_spans = [(chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks]

            tokens = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = tokens["offset_mapping"].squeeze(0)  # (max_len, 2)

            mask = torch.zeros(max_len, dtype=torch.float)
            for start_char, end_char in noun_spans:
                for idx in range(max_len):
                    tok_start, tok_end = offsets[idx].tolist()
                    if tok_start == 0 and tok_end == 0:
                        continue
                    if tok_start >= start_char and tok_end <= end_char:
                        mask[idx] = 1.0

            batch_masks.append(mask)

        return torch.stack(batch_masks)  # (B, max_len)


class SaliencyEstimator(nn.Module):
    """
    Lightweight saliency map estimator for weak region supervision.
    Combines center-prior with learned frequency-based foreground detection.
    Operates on raw image tensors, outputs per-patch saliency scores.
    """
    def __init__(self, grid_size=14):
        super().__init__()
        self.grid_size = grid_size

        # Learnable center-bias (Gaussian prior centered at image center)
        ys = torch.linspace(-1, 1, grid_size)
        xs = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        center_prior = torch.exp(-(xx**2 + yy**2) / 0.5)
        self.register_buffer('center_prior', center_prior.flatten())  # (196,)

        # Lightweight conv to detect foreground from downscaled image
        self.saliency_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        Returns:
            saliency: (B, 196) soft mask, values in [0, 1]
        """
        # Downsample to grid resolution
        x = F.interpolate(images, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
        sal = self.saliency_net(x)  # (B, 1, 14, 14)
        sal = sal.flatten(1)  # (B, 196)

        # Combine learned saliency with center prior
        combined = sal * 0.7 + self.center_prior.unsqueeze(0) * 0.3
        return combined


class GroundingLoss(nn.Module):
    """
    Weak grounding loss with entropy minimization for noun phrase attention.
    """
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, noun_masks, text_mask):
        """
        attn_weights: (B, seq_len, 196) text-to-image cross-attention weights
        noun_masks: (B, seq_len) binary mask for noun phrase tokens
        text_mask: (B, seq_len) attention mask for valid tokens

        Minimizes entropy of noun-phrase attention over patches.
        """
        valid_mask = noun_masks * text_mask.float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return torch.tensor(0.0, device=attn_weights.device)

        attn_safe = attn_weights.clamp(min=1e-8)
        log_attn = attn_safe.log()
        log_attn = log_attn * text_mask.unsqueeze(-1).float()
        attn_safe = attn_safe * text_mask.unsqueeze(-1).float()
        entropy = -(attn_safe * log_attn).sum(dim=-1)

        masked_entropy = (entropy * valid_mask).sum() / num_valid.clamp(min=1)
        max_entropy = torch.log(torch.tensor(196.0, device=attn_weights.device))
        return masked_entropy / max_entropy


class RegionSupervisionLoss(nn.Module):
    """
    Constrains cross-attention to fall within salient/foreground regions.
    Uses saliency maps (or detection boxes) as weak supervision targets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, saliency_map, noun_masks, text_mask):
        """
        attn_weights: (B, seq_len, 196) text-to-image attention
        saliency_map: (B, 196) target region probabilities [0, 1]
        noun_masks: (B, seq_len) noun phrase token indicators
        text_mask: (B, seq_len) valid token mask

        Loss: KL divergence between noun-phrase attention distribution
        and the normalized saliency map (soft target).
        """
        valid_mask = noun_masks * text_mask.float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return torch.tensor(0.0, device=attn_weights.device)

        # Normalize saliency to a probability distribution
        sal_norm = saliency_map / saliency_map.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sal_norm = sal_norm.unsqueeze(1).expand_as(attn_weights)  # (B, seq_len, 196)

        # KL(saliency || attention) for noun phrase tokens only
        attn_safe = attn_weights.clamp(min=1e-8)
        sal_safe = sal_norm.clamp(min=1e-8)
        kl = sal_safe * (sal_safe.log() - attn_safe.log())  # (B, seq_len, 196)
        kl_per_token = kl.sum(dim=-1)  # (B, seq_len)

        # Zero out invalid positions
        kl_per_token = kl_per_token * valid_mask
        loss = kl_per_token.sum() / num_valid.clamp(min=1)

        return loss


class BoxRegionLoss(nn.Module):
    """
    When bounding box annotations are available, constrains attention
    to fall within the box region. Converts boxes to patch-level masks.
    """
    def __init__(self, grid_size=14):
        super().__init__()
        self.grid_size = grid_size

    def boxes_to_patch_mask(self, boxes, device):
        """
        boxes: list of (x1, y1, x2, y2) normalized to [0, 1]
        Returns: (B, 196) binary patch mask
        """
        B = len(boxes)
        masks = torch.zeros(B, self.grid_size * self.grid_size, device=device)

        for i, box in enumerate(boxes):
            if box is None:
                masks[i] = 1.0  # no box = all patches valid
                continue
            x1, y1, x2, y2 = box
            col_start = int(x1 * self.grid_size)
            col_end = max(col_start + 1, int(x2 * self.grid_size))
            row_start = int(y1 * self.grid_size)
            row_end = max(row_start + 1, int(y2 * self.grid_size))

            for r in range(row_start, min(row_end, self.grid_size)):
                for c in range(col_start, min(col_end, self.grid_size)):
                    masks[i, r * self.grid_size + c] = 1.0

        return masks

    def forward(self, attn_weights, box_masks, noun_masks, text_mask):
        """
        attn_weights: (B, seq_len, 196)
        box_masks: (B, 196) binary mask for target region
        noun_masks: (B, seq_len)
        text_mask: (B, seq_len)

        Loss: attention mass outside box regions for noun tokens.
        """
        valid_mask = noun_masks * text_mask.float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return torch.tensor(0.0, device=attn_weights.device)

        # Attention mass outside the box
        outside_mask = 1.0 - box_masks  # (B, 196)
        attn_outside = (attn_weights * outside_mask.unsqueeze(1)).sum(dim=-1)  # (B, seq_len)

        loss = (attn_outside * valid_mask).sum() / num_valid.clamp(min=1)
        return loss
