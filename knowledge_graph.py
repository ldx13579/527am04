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
    """2-layer GCN that embeds knowledge graph nodes into 32 dimensions."""
    def __init__(self, num_nodes=NUM_NODES, input_dim=64, hidden_dim=64, output_dim=32):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.02)
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.register_buffer("adj", build_adjacency_matrix())

    def forward(self):
        """Run GCN on entire graph, return all node embeddings (N, 32)."""
        h = F.relu(self.gcn1(self.node_embeddings, self.adj))
        h = self.gcn2(h, self.adj)
        return h

    def get_embeddings(self, node_indices):
        """Get embeddings for specific node indices. (B, 32) or (B, K, 32)."""
        all_embeds = self.forward()  # (N, 32)
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
