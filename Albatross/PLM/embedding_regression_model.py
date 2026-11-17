"""
Multi-task regression model that takes a PRE-POOLED protein language model (PLM)
embedding (one fixed-length vector per sequence) and predicts 4 properties:
    - asphericity
    - scaled RE
    - scaled RG
    - scaling exponent

No tokenizer, no CLS/PAD tokens, no transformer encoder is needed here because
the PLM embedding already summarizes the whole sequence into one vector.

Architecture
------------
1. (optional) LayerNorm on the raw PLM embedding for stability.
2. Four property branches (small MLPs), each taking the embedding and producing:
       (a) a property-specific "label embedding"
       (b) an auxiliary scalar prediction for that property
3. The (normalized) embedding + all 4 label embeddings are concatenated and
   passed through a fusion network that predicts all 4 properties jointly.
4. Loss = final joint loss + weighted sum of the 4 auxiliary losses.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-property branch 
# ---------------------------------------------------------------------------

class PropertyBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, embed_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embed = self.net(x)
        pred = self.head(embed).squeeze(-1)
        return embed, pred


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class EmbeddingMultiTaskRegressor(nn.Module):

    PROPERTY_NAMES = ["asphericity", "prefactor", "scaled_re", "scaled_rg", "scaling_exponent"]

    def __init__(
        self,
        embedding_dim,
        branch_hidden=128,
        branch_embed=256,
        fusion_hidden=256,
        dropout=0.1,
        normalize_input=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize_input = normalize_input
        if normalize_input:
            self.input_norm = nn.LayerNorm(embedding_dim)

        self.branches = nn.ModuleDict(
            {
                name: PropertyBranch(embedding_dim, branch_hidden, branch_embed, dropout)
                for name in self.PROPERTY_NAMES
            }
        )

        fusion_input_dim = embedding_dim + branch_embed * len(self.PROPERTY_NAMES)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Linear(fusion_hidden // 2, len(self.PROPERTY_NAMES)),
        )

    def forward(self, embedding):
        """
        embedding: [B, embedding_dim] pre-pooled PLM embedding for each sequence
        """
        x = self.input_norm(embedding) if self.normalize_input else embedding

        embeds = []
        aux_preds = {}
        for name in self.PROPERTY_NAMES:
            e, pred = self.branches[name](x)
            embeds.append(e)
            aux_preds[name] = pred

        fusion_input = torch.cat([x] + embeds, dim=-1)
        final_pred = self.fusion(fusion_input)  # [B, 4]

        return {"final_pred": final_pred, "aux_preds": aux_preds}


# ---------------------------------------------------------------------------
# Loss function (identical logic to the sequence-based version)
# ---------------------------------------------------------------------------

def compute_loss(outputs, labels, aux_weight=0.1):
    mse = nn.MSELoss()

    final_loss = mse(outputs["final_pred"], labels)

    aux_loss = 0.0
    for i, name in enumerate(EmbeddingMultiTaskRegressor.PROPERTY_NAMES):
        aux_loss = aux_loss + mse(outputs["aux_preds"][name], labels[:, i])
    aux_loss = aux_loss / len(EmbeddingMultiTaskRegressor.PROPERTY_NAMES)

    total_loss = final_loss + aux_weight * aux_loss
    return total_loss, final_loss, aux_loss
