import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from knowledge_graph import KnowledgeGraphGCN, KGFeatureAggregator, KGAlignmentLoss


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=192, depth=4, heads=3, mlp_ratio=4, shared_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, shared_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_patches=False):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        cls_embed = self.projection(cls_token)
        if return_patches:
            return cls_embed, x[:, 1:]  # (B, 196, vit_dim)
        return cls_embed


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", shared_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, shared_dim)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, input_ids, attention_mask, return_tokens=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        cls_output = last_hidden[:, 0]
        cls_embed = self.projection(cls_output)
        if return_tokens:
            return cls_embed, last_hidden  # (B, seq_len, 768)
        return cls_embed


class CrossAttentionBranch(nn.Module):
    """Single-head cross-attention: query attends to key/value source."""
    def __init__(self, q_dim, kv_dim, shared_dim=256):
        super().__init__()
        self.shared_dim = shared_dim
        self.scale = shared_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, shared_dim)
        self.k_proj = nn.Linear(kv_dim, shared_dim)
        self.v_proj = nn.Linear(kv_dim, shared_dim)
        self.out_proj = nn.Linear(shared_dim, shared_dim)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, query, key_value, q_mask=None, kv_mask=None):
        """
        query: (B, N_q, q_dim)
        key_value: (B, N_kv, kv_dim)
        q_mask: (B, N_q) - 1 for valid, 0 for padding
        kv_mask: (B, N_kv) - 1 for valid, 0 for padding
        Returns:
            pooled: (B, shared_dim)
            attn_weights: (B, N_q, N_kv)
        """
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        Q = self.q_proj(query)      # (B, N_q, shared_dim)
        K = self.k_proj(key_value)  # (B, N_kv, shared_dim)
        V = self.v_proj(key_value)  # (B, N_kv, shared_dim)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, N_q, N_kv)

        # Mask out padded key positions
        if kv_mask is not None:
            attn = attn.masked_fill(kv_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = attn.softmax(dim=-1)  # (B, N_q, N_kv)

        # Clean NaN from fully-masked rows
        if kv_mask is not None:
            nan_rows = attn_weights.isnan().any(dim=-1, keepdim=True)
            attn_weights = attn_weights.masked_fill(nan_rows, 0.0)

        attended = attn_weights @ V  # (B, N_q, shared_dim)

        # Pool over query dimension with mask
        if q_mask is not None:
            mask_exp = q_mask.unsqueeze(-1).float()
            attended = attended * mask_exp
            pooled = attended.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            pooled = attended.mean(dim=1)

        pooled = self.out_proj(pooled)
        return pooled, attn_weights


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention: image↔text both get fused features."""
    def __init__(self, img_dim, text_dim, shared_dim=256):
        super().__init__()
        # Text→Image: text queries attend to image patches (for image-side fusion)
        self.text_to_img = CrossAttentionBranch(
            q_dim=text_dim, kv_dim=img_dim, shared_dim=shared_dim
        )
        # Image→Text: image patches attend to text tokens (for text-side fusion)
        self.img_to_text = CrossAttentionBranch(
            q_dim=img_dim, kv_dim=text_dim, shared_dim=shared_dim
        )

    def forward(self, img_patches, text_tokens, text_mask=None):
        """
        img_patches: (B, 196, img_dim)
        text_tokens: (B, seq_len, text_dim)
        text_mask: (B, seq_len)
        Returns:
            img_fused: (B, shared_dim) - image-side fusion from text→image attention
            txt_fused: (B, shared_dim) - text-side fusion from image→text attention
            t2i_attn: (B, seq_len, 196) - text-to-image attention weights
            i2t_attn: (B, 196, seq_len) - image-to-text attention weights
        """
        # Text queries attend to image patches → image-side fused feature
        img_fused, t2i_attn = self.text_to_img(
            query=text_tokens, key_value=img_patches,
            q_mask=text_mask, kv_mask=None
        )
        # Image patches attend to text tokens → text-side fused feature
        txt_fused, i2t_attn = self.img_to_text(
            query=img_patches, key_value=text_tokens,
            q_mask=None, kv_mask=text_mask
        )
        return img_fused, txt_fused, t2i_attn, i2t_attn


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ViTTiny(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.vit_dim,
            depth=config.vit_layers,
            heads=config.vit_heads,
            mlp_ratio=config.vit_mlp_ratio,
            shared_dim=config.shared_dim,
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            shared_dim=config.shared_dim,
        )
        self.cross_attention = BidirectionalCrossAttention(
            img_dim=config.vit_dim,
            text_dim=config.text_hidden_dim,
            shared_dim=config.shared_dim,
        )
        self.img_gate = nn.Parameter(torch.tensor(0.0))
        self.txt_gate = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.init_temperature))
        )

    @property
    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=100.0)

    def encode_image(self, images):
        """Unified interface: returns L2-normalized image embeddings (no cross-attention)."""
        img_cls = self.image_encoder(images, return_patches=False)
        return F.normalize(img_cls, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """Unified interface: returns L2-normalized text embeddings (no cross-attention)."""
        txt_cls = self.text_encoder(input_ids, attention_mask, return_tokens=False)
        return F.normalize(txt_cls, dim=-1)

    def encode_pair(self, images, input_ids, attention_mask):
        """Paired encoding: both sides go through cross-attention fusion then normalize."""
        img_cls, img_patches = self.image_encoder(images, return_patches=True)
        txt_cls, txt_tokens = self.text_encoder(input_ids, attention_mask, return_tokens=True)

        img_fused, txt_fused, _, _ = self.cross_attention(
            img_patches, txt_tokens, text_mask=attention_mask
        )

        img_g = torch.sigmoid(self.img_gate)
        image_embeds = F.normalize(img_cls * (1 - img_g) + img_fused * img_g, dim=-1)

        txt_g = torch.sigmoid(self.txt_gate)
        text_embeds = F.normalize(txt_cls * (1 - txt_g) + txt_fused * txt_g, dim=-1)

        return image_embeds, text_embeds

    def forward(self, images, input_ids, attention_mask, return_cross_attn=False):
        """Full forward with bidirectional cross-attention fusion."""
        img_cls, img_patches = self.image_encoder(images, return_patches=True)
        txt_cls, txt_tokens = self.text_encoder(input_ids, attention_mask, return_tokens=True)

        img_fused, txt_fused, t2i_attn, i2t_attn = self.cross_attention(
            img_patches, txt_tokens, text_mask=attention_mask
        )

        # Gated fusion for image side
        img_g = torch.sigmoid(self.img_gate)
        image_embeds = F.normalize(img_cls * (1 - img_g) + img_fused * img_g, dim=-1)

        # Gated fusion for text side
        txt_g = torch.sigmoid(self.txt_gate)
        text_embeds = F.normalize(txt_cls * (1 - txt_g) + txt_fused * txt_g, dim=-1)

        if return_cross_attn:
            return image_embeds, text_embeds, self.temperature, t2i_attn, i2t_attn
        return image_embeds, text_embeds, self.temperature


class ModalityPredictor(nn.Module):
    """2-layer MLP + single-head cross-attention for cross-modal reconstruction."""
    def __init__(self, shared_dim=256, hidden_ratio=2):
        super().__init__()
        hidden_dim = shared_dim * hidden_ratio
        # MLP pathway
        self.mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shared_dim),
        )
        # Lightweight single-head self-refinement attention
        self.attn_norm = nn.LayerNorm(shared_dim)
        self.q_proj = nn.Linear(shared_dim, shared_dim)
        self.k_proj = nn.Linear(shared_dim, shared_dim)
        self.v_proj = nn.Linear(shared_dim, shared_dim)
        self.out_proj = nn.Linear(shared_dim, shared_dim)
        self.scale = shared_dim ** -0.5
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, shared_dim) -> (B, shared_dim)"""
        h = self.mlp(x)
        # Self-refinement: treat as single-token sequence
        h_norm = self.attn_norm(h)
        q = self.q_proj(h_norm).unsqueeze(1)  # (B, 1, D)
        k = self.k_proj(h_norm).unsqueeze(1)
        v = self.v_proj(h_norm).unsqueeze(1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        refined = (attn @ v).squeeze(1)
        h = h + self.out_proj(refined)
        return h


class ModalityDropout(nn.Module):
    """Modality missing training with MLP+attention predictors."""
    def __init__(self, shared_dim=256, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.img_to_txt_predictor = ModalityPredictor(shared_dim)
        self.txt_to_img_predictor = ModalityPredictor(shared_dim)

    def forward(self, img_cls, txt_cls):
        """
        During training, with drop_prob probability, drop one modality and
        predict it from the other.
        Returns:
            img_feat: (B, shared_dim) - image features (possibly predicted)
            txt_feat: (B, shared_dim) - text features (possibly predicted)
            recon_loss: scalar reconstruction loss
        """
        if not self.training:
            return img_cls, txt_cls, torch.tensor(0.0, device=img_cls.device)

        B = img_cls.shape[0]
        device = img_cls.device
        rand = torch.rand(B, device=device)

        # Masks: which samples drop image, which drop text
        drop_img_mask = (rand < self.drop_prob).float().unsqueeze(-1)  # (B, 1)
        drop_txt_mask = ((rand >= self.drop_prob) & (rand < 2 * self.drop_prob)).float().unsqueeze(-1)

        # Predict missing modality
        predicted_img = self.txt_to_img_predictor(txt_cls)
        predicted_txt = self.img_to_txt_predictor(img_cls)

        # Replace dropped modality with prediction
        img_feat = img_cls * (1 - drop_img_mask) + predicted_img * drop_img_mask
        txt_feat = txt_cls * (1 - drop_txt_mask) + predicted_txt * drop_txt_mask

        # Reconstruction loss: MSE between predicted and actual for dropped samples
        recon_loss = torch.tensor(0.0, device=device)
        n_dropped = 0
        if drop_img_mask.sum() > 0:
            recon_loss = recon_loss + F.mse_loss(
                predicted_img * drop_img_mask,
                img_cls.detach() * drop_img_mask,
                reduction='sum'
            )
            n_dropped += drop_img_mask.sum()
        if drop_txt_mask.sum() > 0:
            recon_loss = recon_loss + F.mse_loss(
                predicted_txt * drop_txt_mask,
                txt_cls.detach() * drop_txt_mask,
                reduction='sum'
            )
            n_dropped += drop_txt_mask.sum()
        if n_dropped > 0:
            recon_loss = recon_loss / n_dropped

        return img_feat, txt_feat, recon_loss


class KGEnhancedCLIPModel(nn.Module):
    """CLIPModel enhanced with Knowledge Graph GCN embeddings and modality dropout."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        kg_dim = getattr(config, 'kg_dim', 32)
        drop_prob = getattr(config, 'modality_drop_prob', 0.1)

        self.image_encoder = ViTTiny(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.vit_dim,
            depth=config.vit_layers,
            heads=config.vit_heads,
            mlp_ratio=config.vit_mlp_ratio,
            shared_dim=config.shared_dim,
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            shared_dim=config.shared_dim,
        )
        self.cross_attention = BidirectionalCrossAttention(
            img_dim=config.vit_dim,
            text_dim=config.text_hidden_dim,
            shared_dim=config.shared_dim,
        )

        # Knowledge Graph GCN (pretrained init from DistilBERT)
        self.kg_gcn = KnowledgeGraphGCN(
            output_dim=kg_dim,
            use_pretrained_init=True,
            text_model_name=config.text_model_name,
        )
        self.kg_aggregator = KGFeatureAggregator(kg_dim=kg_dim)

        # KG-text contrastive alignment
        self.kg_alignment = KGAlignmentLoss(
            kg_dim=kg_dim, shared_dim=config.shared_dim
        )

        # Fusion: project (shared_dim + kg_dim) back to shared_dim
        fused_dim = config.shared_dim + kg_dim
        self.img_kg_fusion = nn.Linear(fused_dim, config.shared_dim)
        self.txt_kg_fusion = nn.Linear(fused_dim, config.shared_dim)
        nn.init.xavier_uniform_(self.img_kg_fusion.weight)
        nn.init.zeros_(self.img_kg_fusion.bias)
        nn.init.xavier_uniform_(self.txt_kg_fusion.weight)
        nn.init.zeros_(self.txt_kg_fusion.bias)

        # Modality dropout (MLP + attention predictor)
        self.modality_dropout = ModalityDropout(
            shared_dim=config.shared_dim, drop_prob=drop_prob
        )

        self.img_gate = nn.Parameter(torch.tensor(0.0))
        self.txt_gate = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.init_temperature))
        )

    @property
    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=100.0)

    def _fuse_with_kg(self, img_embeds, txt_embeds, kg_node_indices, kg_node_mask):
        """Concatenate KG embeddings with image/text features."""
        kg_embeds = self.kg_gcn.get_embeddings(kg_node_indices)  # (B, max_nodes, 32)
        kg_pooled = self.kg_aggregator(kg_embeds, kg_node_mask)  # (B, 32)

        img_with_kg = torch.cat([img_embeds, kg_pooled], dim=-1)  # (B, shared_dim+32)
        txt_with_kg = torch.cat([txt_embeds, kg_pooled], dim=-1)

        img_fused = self.img_kg_fusion(img_with_kg)  # (B, shared_dim)
        txt_fused = self.txt_kg_fusion(txt_with_kg)

        return img_fused, txt_fused, kg_embeds

    def encode_image(self, images, kg_node_indices=None, kg_node_mask=None):
        """Encode images with optional KG fusion. Active at both train and test time."""
        img_cls = self.image_encoder(images, return_patches=False)
        if kg_node_indices is not None and kg_node_mask is not None:
            kg_embeds = self.kg_gcn.get_embeddings(kg_node_indices)
            kg_pooled = self.kg_aggregator(kg_embeds, kg_node_mask)
            img_with_kg = torch.cat([img_cls, kg_pooled], dim=-1)
            img_cls = self.img_kg_fusion(img_with_kg)
        return F.normalize(img_cls, dim=-1)

    def encode_text(self, input_ids, attention_mask, kg_node_indices=None, kg_node_mask=None):
        """Encode text with optional KG fusion. Active at both train and test time."""
        txt_cls = self.text_encoder(input_ids, attention_mask, return_tokens=False)
        if kg_node_indices is not None and kg_node_mask is not None:
            kg_embeds = self.kg_gcn.get_embeddings(kg_node_indices)
            kg_pooled = self.kg_aggregator(kg_embeds, kg_node_mask)
            txt_with_kg = torch.cat([txt_cls, kg_pooled], dim=-1)
            txt_cls = self.txt_kg_fusion(txt_with_kg)
        return F.normalize(txt_cls, dim=-1)

    def encode_pair(self, images, input_ids, attention_mask,
                    kg_node_indices=None, kg_node_mask=None):
        """Paired encoding with cross-attention and optional KG fusion."""
        img_cls, img_patches = self.image_encoder(images, return_patches=True)
        txt_cls, txt_tokens = self.text_encoder(input_ids, attention_mask, return_tokens=True)

        img_fused, txt_fused, _, _ = self.cross_attention(
            img_patches, txt_tokens, text_mask=attention_mask
        )

        img_g = torch.sigmoid(self.img_gate)
        image_embeds = img_cls * (1 - img_g) + img_fused * img_g

        txt_g = torch.sigmoid(self.txt_gate)
        text_embeds = txt_cls * (1 - txt_g) + txt_fused * txt_g

        if kg_node_indices is not None and kg_node_mask is not None:
            kg_embeds = self.kg_gcn.get_embeddings(kg_node_indices)
            kg_pooled = self.kg_aggregator(kg_embeds, kg_node_mask)
            image_embeds = self.img_kg_fusion(torch.cat([image_embeds, kg_pooled], dim=-1))
            text_embeds = self.txt_kg_fusion(torch.cat([text_embeds, kg_pooled], dim=-1))

        return F.normalize(image_embeds, dim=-1), F.normalize(text_embeds, dim=-1)

    def forward(self, images, input_ids, attention_mask, kg_node_indices=None,
                kg_node_mask=None, return_cross_attn=False):
        """
        Full forward with KG fusion, modality dropout, and KG alignment loss.
        kg_node_indices: (B, max_nodes) - indices into KG
        kg_node_mask: (B, max_nodes) - 1 for valid nodes
        """
        img_cls, img_patches = self.image_encoder(images, return_patches=True)
        txt_cls, txt_tokens = self.text_encoder(input_ids, attention_mask, return_tokens=True)

        # Cross-attention fusion
        img_fused, txt_fused, t2i_attn, i2t_attn = self.cross_attention(
            img_patches, txt_tokens, text_mask=attention_mask
        )

        # Gated fusion
        img_g = torch.sigmoid(self.img_gate)
        image_embeds = img_cls * (1 - img_g) + img_fused * img_g

        txt_g = torch.sigmoid(self.txt_gate)
        text_embeds = txt_cls * (1 - txt_g) + txt_fused * txt_g

        # KG fusion + alignment loss
        kg_align_loss = torch.tensor(0.0, device=images.device)
        if kg_node_indices is not None and kg_node_mask is not None:
            image_embeds, text_embeds, kg_embeds = self._fuse_with_kg(
                image_embeds, text_embeds, kg_node_indices, kg_node_mask
            )
            # Contrastive alignment between KG embeddings and text features
            kg_align_loss = self.kg_alignment(kg_embeds, txt_cls, kg_node_mask)

        # Modality dropout (training only)
        image_embeds, text_embeds, recon_loss = self.modality_dropout(
            image_embeds, text_embeds
        )

        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        if return_cross_attn:
            return image_embeds, text_embeds, self.temperature, t2i_attn, i2t_attn, recon_loss, kg_align_loss
        return image_embeds, text_embeds, self.temperature, recon_loss, kg_align_loss


class BaselineCLIPModel(nn.Module):
    """CLIP model without cross-attention — same unified interface."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = ViTTiny(
            img_size=config.img_size,
            patch_size=config.patch_size,
            dim=config.vit_dim,
            depth=config.vit_layers,
            heads=config.vit_heads,
            mlp_ratio=config.vit_mlp_ratio,
            shared_dim=config.shared_dim,
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            shared_dim=config.shared_dim,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.init_temperature))
        )

    @property
    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=100.0)

    def encode_image(self, images):
        """Unified interface: returns L2-normalized image embeddings."""
        img_cls = self.image_encoder(images, return_patches=False)
        return F.normalize(img_cls, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """Unified interface: returns L2-normalized text embeddings."""
        txt_cls = self.text_encoder(input_ids, attention_mask, return_tokens=False)
        return F.normalize(txt_cls, dim=-1)

    def encode_pair(self, images, input_ids, attention_mask):
        """Baseline has no cross-attention, so encode_pair == independent encoding."""
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds, self.temperature
