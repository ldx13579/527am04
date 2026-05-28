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


class GroundingLoss(nn.Module):
    """
    Weak grounding loss: encourages attention on noun-phrase tokens to be
    spatially concentrated (low entropy) on image patches.
    """
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, noun_masks, text_mask):
        """
        attn_weights: (B, seq_len, 196) cross-attention weights
        noun_masks: (B, seq_len) binary mask for noun phrase tokens
        text_mask: (B, seq_len) attention mask for valid tokens

        Loss: for noun-phrase tokens, minimize entropy of their attention
        distribution over image patches (encourages spatial focus).
        """
        valid_mask = noun_masks * text_mask.float()  # (B, seq_len)
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return torch.tensor(0.0, device=attn_weights.device)

        # Only compute entropy for valid (non-padded) tokens to avoid NaN
        attn_safe = attn_weights.clamp(min=1e-8)
        log_attn = attn_safe.log()
        # Zero out padded rows to prevent NaN propagation
        log_attn = log_attn * text_mask.unsqueeze(-1).float()
        attn_safe = attn_safe * text_mask.unsqueeze(-1).float()
        entropy = -(attn_safe * log_attn).sum(dim=-1)  # (B, seq_len)

        masked_entropy = (entropy * valid_mask).sum() / num_valid.clamp(min=1)
        max_entropy = torch.log(torch.tensor(196.0, device=attn_weights.device))
        normalized_loss = masked_entropy / max_entropy

        return normalized_loss
