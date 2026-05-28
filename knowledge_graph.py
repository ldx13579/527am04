import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# ============================================================
# Knowledge Graph: 200 common objects with parent-child taxonomy
# ============================================================

TAXONOMY = {
    "animal": [
        "cat", "dog", "horse", "cow", "sheep", "pig", "chicken", "duck",
        "bird", "fish", "rabbit", "mouse", "elephant", "lion", "tiger",
        "bear", "deer", "wolf", "fox", "monkey", "zebra", "giraffe",
        "penguin", "eagle", "owl", "parrot", "snake", "frog", "turtle",
        "whale", "dolphin", "shark", "crab", "butterfly", "bee", "ant",
        "spider", "squirrel", "goat", "donkey",
    ],
    "vehicle": [
        "car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane",
        "helicopter", "boat", "ship", "subway", "taxi", "ambulance",
        "fire truck", "van", "scooter", "skateboard", "canoe", "yacht", "tractor",
    ],
    "food": [
        "apple", "banana", "orange", "grape", "strawberry", "watermelon",
        "pizza", "hamburger", "sandwich", "cake", "bread", "rice", "pasta",
        "salad", "soup", "ice cream", "cookie", "chocolate", "cheese", "egg",
    ],
    "furniture": [
        "chair", "table", "sofa", "bed", "desk", "bookshelf", "cabinet",
        "dresser", "stool", "bench", "couch", "wardrobe", "nightstand",
        "ottoman", "recliner",
    ],
    "electronics": [
        "computer", "phone", "laptop", "tablet", "television", "camera",
        "keyboard", "monitor", "printer", "speaker", "headphones",
        "microphone", "router", "clock", "watch",
    ],
    "clothing": [
        "shirt", "pants", "dress", "jacket", "hat", "shoes", "socks",
        "gloves", "scarf", "tie", "coat", "boots", "sandals", "sweater",
        "skirt",
    ],
    "sports equipment": [
        "ball", "bat", "racket", "net", "helmet", "glove", "surfboard",
        "ski", "snowboard", "frisbee",
    ],
    "kitchen item": [
        "knife", "fork", "spoon", "plate", "cup", "bowl", "pot",
        "pan", "oven", "refrigerator",
    ],
    "outdoor object": [
        "tree", "flower", "grass", "rock", "mountain", "river", "lake",
        "bridge", "fence", "bench",
    ],
    "building": [
        "house", "church", "tower", "castle", "school", "hospital",
        "store", "restaurant", "hotel", "garage",
    ],
    "tool": [
        "hammer", "screwdriver", "wrench", "saw", "drill", "scissors",
        "shovel", "axe", "ladder", "rope",
    ],
    "person": [
        "man", "woman", "child", "boy", "girl", "baby", "people",
        "player", "rider", "worker",
    ],
}

# Build node list and edges
ALL_NODES = []
NODE_TO_IDX = {}
EDGES = []


def _build_graph():
    global ALL_NODES, NODE_TO_IDX, EDGES
    idx = 0
    # Add parent categories first
    for parent in TAXONOMY:
        ALL_NODES.append(parent)
        NODE_TO_IDX[parent] = idx
        idx += 1
    # Add child nodes
    for parent, children in TAXONOMY.items():
        for child in children:
            if child not in NODE_TO_IDX:
                ALL_NODES.append(child)
                NODE_TO_IDX[child] = idx
                idx += 1
    # Build edges (bidirectional parent-child)
    for parent, children in TAXONOMY.items():
        p_idx = NODE_TO_IDX[parent]
        for child in children:
            c_idx = NODE_TO_IDX[child]
            EDGES.append((p_idx, c_idx))
            EDGES.append((c_idx, p_idx))
    # Self-loops
    for i in range(len(ALL_NODES)):
        EDGES.append((i, i))


_build_graph()
NUM_NODES = len(ALL_NODES)


def build_adjacency_matrix():
    adj = torch.zeros(NUM_NODES, NUM_NODES)
    for (i, j) in EDGES:
        adj[i, j] = 1.0
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1).clamp(min=1)
    d_inv_sqrt = degree.pow(-0.5)
    adj_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
    return adj_norm


# ============================================================
# 2-Layer GCN
# ============================================================

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        # x: (N, in_dim), adj: (N, N) normalized adjacency
        h = self.linear(x)
        h = adj @ h
        return h


class KnowledgeGraphGCN(nn.Module):
    """2-layer GCN that embeds knowledge graph nodes into 32 dimensions.
    Supports pretrained word vector initialization via DistilBERT."""
    def __init__(self, num_nodes=NUM_NODES, input_dim=64, hidden_dim=64, output_dim=32,
                 use_pretrained_init=True, text_model_name="distilbert-base-uncased"):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        if use_pretrained_init:
            init_embeds = self._get_pretrained_embeddings(text_model_name, input_dim)
            self.node_embeddings = nn.Parameter(init_embeds)
        else:
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.02)

        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.register_buffer("adj", build_adjacency_matrix())

    @staticmethod
    @torch.no_grad()
    def _get_pretrained_embeddings(model_name, target_dim):
        """Extract word embeddings from DistilBERT for each KG node name,
        then project to target_dim via PCA-like linear projection."""
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings = []
        for node_name in ALL_NODES:
            tokens = tokenizer(node_name, return_tensors="pt", padding=False, truncation=True)
            outputs = model(**tokens)
            # Mean-pool all token embeddings for multi-word nodes
            node_emb = outputs.last_hidden_state[0].mean(dim=0)  # (768,)
            embeddings.append(node_emb)

        emb_matrix = torch.stack(embeddings, dim=0)  # (N, 768)
        # Project to target_dim using SVD (keep top components)
        U, S, V = torch.svd(emb_matrix - emb_matrix.mean(dim=0, keepdim=True))
        projected = U[:, :target_dim] * S[:target_dim].unsqueeze(0)  # (N, target_dim)
        # Normalize to reasonable scale
        projected = projected / projected.std() * 0.02
        return projected

    def forward(self):
        """Run GCN on entire graph, return all node embeddings (N, output_dim)."""
        h = F.relu(self.gcn1(self.node_embeddings, self.adj))
        h = self.gcn2(h, self.adj)
        return h

    def get_embeddings(self, node_indices):
        """Get embeddings for specific node indices. (B, K, output_dim)."""
        all_embeds = self.forward()  # (N, output_dim)
        return all_embeds[node_indices]


# ============================================================
# Caption -> Knowledge Graph node matching
# ============================================================

# Precompute sorted nodes by length (longest first for greedy matching)
_SORTED_NODES = sorted(NODE_TO_IDX.keys(), key=lambda x: len(x), reverse=True)
_NODE_PATTERNS = [(node, re.compile(r'\b' + re.escape(node) + r'\b', re.IGNORECASE))
                  for node in _SORTED_NODES]


def extract_kg_nodes_from_caption(caption):
    """Extract knowledge graph node indices from a caption string.
    Returns list of matched node indices."""
    matched = []
    caption_lower = caption.lower()
    for node, pattern in _NODE_PATTERNS:
        if pattern.search(caption_lower):
            matched.append(NODE_TO_IDX[node])
    return matched


def extract_kg_nodes_batch(captions, max_nodes=5):
    """Extract KG node indices for a batch of captions.
    Returns:
        node_indices: (B, max_nodes) padded with 0
        node_mask: (B, max_nodes) 1 for valid, 0 for padding
    """
    B = len(captions)
    node_indices = torch.zeros(B, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.float)

    for i, caption in enumerate(captions):
        nodes = extract_kg_nodes_from_caption(caption)
        n = min(len(nodes), max_nodes)
        if n > 0:
            node_indices[i, :n] = torch.tensor(nodes[:n])
            node_mask[i, :n] = 1.0

    return node_indices, node_mask


class KGFeatureAggregator(nn.Module):
    """Aggregate KG node embeddings into a single vector for each sample."""
    def __init__(self, kg_dim=32):
        super().__init__()
        self.kg_dim = kg_dim

    def forward(self, kg_embeds, node_mask):
        """
        kg_embeds: (B, max_nodes, 32)
        node_mask: (B, max_nodes)
        Returns: (B, 32) mean-pooled KG embedding (zero if no nodes matched)
        """
        mask_exp = node_mask.unsqueeze(-1)  # (B, max_nodes, 1)
        masked_embeds = kg_embeds * mask_exp
        denom = mask_exp.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = masked_embeds.sum(dim=1) / denom  # (B, 32)
        return pooled


class KGAlignmentLoss(nn.Module):
    """Contrastive loss to align GCN node embeddings with text encoder representations.
    Pulls together GCN embedding and text embedding of the same concept,
    pushes apart embeddings of different concepts."""
    def __init__(self, kg_dim=32, shared_dim=256, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.kg_proj = nn.Linear(kg_dim, shared_dim)
        nn.init.xavier_uniform_(self.kg_proj.weight)
        nn.init.zeros_(self.kg_proj.bias)

    def forward(self, kg_embeds, text_embeds, node_mask):
        """
        kg_embeds: (B, max_nodes, kg_dim) - GCN embeddings for matched nodes
        text_embeds: (B, shared_dim) - text encoder CLS embeddings
        node_mask: (B, max_nodes) - 1 for valid nodes
        Returns: scalar contrastive alignment loss
        """
        # Pool KG embeddings
        mask_exp = node_mask.unsqueeze(-1)
        kg_pooled = (kg_embeds * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        # Only compute on samples that have KG nodes
        has_nodes = node_mask.sum(dim=1) > 0  # (B,)
        if has_nodes.sum() == 0:
            return torch.tensor(0.0, device=kg_embeds.device)

        kg_pooled = kg_pooled[has_nodes]  # (N, kg_dim)
        text_embeds = text_embeds[has_nodes]  # (N, shared_dim)

        # Project KG to shared space
        kg_projected = F.normalize(self.kg_proj(kg_pooled), dim=-1)  # (N, shared_dim)
        text_normed = F.normalize(text_embeds, dim=-1)  # (N, shared_dim)

        # InfoNCE: each KG embedding should be close to its corresponding text embedding
        N = kg_projected.shape[0]
        sim = (kg_projected @ text_normed.T) / self.temperature  # (N, N)
        labels = torch.arange(N, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss
