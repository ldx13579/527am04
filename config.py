from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    # ViT-Tiny (from scratch)
    img_size: int = 224
    patch_size: int = 16
    vit_layers: int = 4
    vit_dim: int = 192
    vit_heads: int = 3
    vit_mlp_ratio: int = 4

    # Text encoder (DistilBERT pretrained)
    text_model_name: str = "distilbert-base-uncased"
    text_hidden_dim: int = 768
    max_text_len: int = 64

    # Shared embedding
    shared_dim: int = 256

    # Training
    total_steps: int = 20000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Loss
    init_temperature: float = 0.07
    hard_negative_weight: float = 0.5
    grounding_weight: float = 0.3
    region_weight: float = 0.2
    recon_weight: float = 0.1

    # Knowledge Graph
    kg_dim: int = 32
    kg_max_nodes: int = 5
    modality_drop_prob: float = 0.1
    kg_align_weight_min: float = 0.05
    kg_align_weight_max: float = 0.25
    img_kg_cls_weight: float = 0.1

    # Evaluation
    eval_every: int = 2000

    # Hardware
    use_amp: bool = True
    num_workers: int = 4
    seed: int = 42
