import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


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

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return F.normalize(self.projection(cls_token), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", shared_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, shared_dim)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return F.normalize(self.projection(cls_output), dim=-1)


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        return image_embeds, text_embeds, self.temperature
