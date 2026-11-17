import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

SEQ_LEN = 501

CHUNK_SIZE = 25
STRIDE = 15

DIM = 512

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "hierarchical_protein_model_cs=50_stride=25"

INPUT_FILE = "ALBATROSS_SEQ_500/all_test_sequences.txt"







SAVE_DIR = "ALBATROSS_SEQ_500/all_albatross_test_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD VOCAB
# ============================================================

with open(
    os.path.join(MODEL_DIR, "vocab.json")
) as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]

PAD_ID = vocab_data["PAD_ID"]

itos = {v: k for k, v in stoi.items()}

VOCAB_SIZE = len(stoi)

PAD_TOKEN = "[PAD]"

print("Vocab loaded")

# ============================================================
# CHUNKING
# ============================================================

#def create_chunks(x):

#    chunks = []

#    for i in range(
#        0,
#        SEQ_LEN - CHUNK_SIZE + 1,
#        STRIDE
#    ):
#        chunks.append(
#            x[:, i:i+CHUNK_SIZE]
#        )

#    return torch.stack(
#        chunks,
#        dim=1
#    )



def create_chunks(x):

    starts = list(
        range(
            0,
            SEQ_LEN - CHUNK_SIZE + 1,
            STRIDE
        )
    )

    if starts[-1] != SEQ_LEN - CHUNK_SIZE:
        starts.append(
            SEQ_LEN - CHUNK_SIZE
        )

    chunks = [
        x[:, s:s+CHUNK_SIZE]
        for s in starts
    ]

    return torch.stack(
        chunks,
        dim=1
    )

# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dim,
        max_len=512
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            dim
        )

        pos = torch.arange(
            max_len
        ).unsqueeze(1)

        div = torch.exp(
            torch.arange(
                0,
                dim,
                2
            ) *
            (
                -math.log(10000.0)
                / dim
            )
        )

        pe[:, 0::2] = torch.sin(
            pos * div
        )

        pe[:, 1::2] = torch.cos(
            pos * div
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return (
            x +
            self.pe[:, :x.size(1)]
        )

# ============================================================
# LOCAL ENCODER
# ============================================================

class LocalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(
            VOCAB_SIZE,
            DIM,
            padding_idx=PAD_ID
        )

        self.pos = PositionalEncoding(
            DIM,
            CHUNK_SIZE + 1
        )

        self.cls = nn.Parameter(
            torch.randn(
                1,
                1,
                DIM
            )
        )

        enc = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            enc,
            num_layers=2
        )

        self.norm = nn.LayerNorm(
            DIM
        )

        self.lm_head = nn.Linear(
            DIM,
            VOCAB_SIZE
        )

    def forward(self, x):

        B, L = x.shape

        h = self.embed(x)

        cls = self.cls.expand(
            B,
            -1,
            -1
        )

        h = torch.cat(
            [cls, h],
            dim=1
        )

        h = self.pos(h)

        pad_mask = (x == PAD_ID)

        pad_mask = torch.cat(
            [
                torch.zeros(
                    B,
                    1,
                    dtype=torch.bool,
                    device=x.device
                ),
                pad_mask
            ],
            dim=1
        )

        h = self.encoder(
            h,
            src_key_padding_mask=pad_mask
        )

        h = self.norm(h)

        cls_token = h[:, 0]

        token_repr = h[:, 1:]

        logits = self.lm_head(
            token_repr
        )

        return cls_token, logits

# ============================================================
# GLOBAL ENCODER
# ============================================================

class GlobalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.global_cls = nn.Parameter(
            torch.randn(
                1,
                1,
                DIM
            )
        )

        self.pos = PositionalEncoding(
            DIM,
            512
        )

        enc = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            enc,
            num_layers=2
        )

        self.norm = nn.LayerNorm(
            DIM
        )

    def forward(
        self,
        chunk_embeddings,
        chunk_valid
    ):

        B, N, D = chunk_embeddings.shape

        cls = self.global_cls.expand(
            B,
            -1,
            -1
        )

        h = torch.cat(
            [cls, chunk_embeddings],
            dim=1
        )

        h = self.pos(h)

        global_pad_mask = torch.cat(
            [
                torch.zeros(
                    B,
                    1,
                    dtype=torch.bool,
                    device=h.device
                ),
                ~chunk_valid
            ],
            dim=1
        )

        h = self.encoder(
            h,
            src_key_padding_mask=global_pad_mask
        )

        h = self.norm(h)

        return h[:, 0]

# ============================================================
# FULL MODEL
# ============================================================

class HierarchicalModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.local = LocalEncoder()

        self.global_enc = GlobalEncoder()

    def forward(self, chunks):

        B, N, C = chunks.shape

        cls_tokens = []

        logits_list = []

        chunk_valid = (
            (chunks != PAD_ID)
            .sum(dim=2)
            >= 4
        )

        for i in range(N):

            cls_token, logits = self.local(
                chunks[:, i]
            )

            cls_tokens.append(
                cls_token
            )

            logits_list.append(
                logits
            )

        cls_tokens = torch.stack(
            cls_tokens,
            dim=1
        )

        z = self.global_enc(
            cls_tokens,
            chunk_valid
        )

        #return z, logits_list
        return z, cls_tokens, logits_list

# ============================================================
# LOAD MODEL
# ============================================================

model = HierarchicalModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "model.pt"
        ),
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded")

# ============================================================
# ENCODER
# ============================================================

def encode_sequence(seq):

    seq = seq.strip().upper()

    tokens = [
        stoi.get(a, stoi["X"])
        for a in seq
    ]

    length = min(
        len(tokens),
        SEQ_LEN
    )

    if len(tokens) > SEQ_LEN:
        tokens = tokens[:SEQ_LEN]
    else:
        tokens += [
            PAD_ID
        ] * (
            SEQ_LEN -
            len(tokens)
        )

    return (
        torch.tensor(tokens),
        length
    )

# ============================================================
# LOAD DATA
# ============================================================

sequences = []

with open(INPUT_FILE) as f:

    for line in f:

        line = line.strip()

        if line:
            sequences.append(
                line
            )


def mask_sequence(x, p=0.4):
    x = x.clone()

    valid = (x != PAD_ID)

    mask = (torch.rand_like(x.float()) < p) & valid

    # replace with X token (or PAD if you prefer)
    x[mask] = stoi["X"]

    return x
    
    
    

# ============================================================
# OUTPUT STORAGE
# ============================================================

comparison = []

global_embeddings = []
local_embeddings = []

overall_correct = 0
overall_total = 0

# ============================================================
# MAIN LOOP
# ============================================================

for seq in sequences:

    x, length = encode_sequence(seq)
    x = x.unsqueeze(0).to(DEVICE)
    x = mask_sequence(x, p=0.2)

    chunks = create_chunks(x)

    with torch.no_grad():
        z, cls_tokens, logits_list = model(chunks)

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    local_emb = cls_tokens.mean(dim=1).squeeze(0).cpu().numpy().tolist()
    global_emb = z.squeeze(0).cpu().numpy().tolist()

    local_embeddings.append(local_emb)
    global_embeddings.append(global_emb)

    # ========================================================
    # PROBABILISTIC VOTING FOR RECONSTRUCTION
    # ========================================================

    VOCAB_SIZE = len(stoi)

    vote = np.zeros((SEQ_LEN, VOCAB_SIZE), dtype=np.float32)
    count = np.zeros((SEQ_LEN,), dtype=np.float32)

    starts = list(range(0, SEQ_LEN - CHUNK_SIZE + 1, STRIDE))
    if starts[-1] != SEQ_LEN - CHUNK_SIZE:
        starts.append(SEQ_LEN - CHUNK_SIZE)

    for chunk_idx, logits in enumerate(logits_list):

        prob = F.softmax(logits, dim=-1)[0].cpu().numpy()

        start = starts[chunk_idx]

        for j in range(CHUNK_SIZE):

            pos = start + j

            if pos >= length:
                continue

            vote[pos] += prob[j]
            count[pos] += 1.0

    # ========================================================
    # FINAL RECONSTRUCTION
    # ========================================================

    recon_tokens = np.full(SEQ_LEN, PAD_ID, dtype=np.int64)

    for i in range(length):

        if count[i] > 0:
            recon_tokens[i] = np.argmax(vote[i] / count[i])
        else:
            recon_tokens[i] = PAD_ID

    recon_seq = "".join(
        itos[int(tok)]
        for tok in recon_tokens[:length]
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    L = min(len(seq), len(recon_seq))

    correct = sum(seq[i] == recon_seq[i] for i in range(L))

    overall_correct += correct
    overall_total += L

    acc = correct / L

    # ========================================================
    # STORE OUTPUT
    # ========================================================

    comparison.append(f"ORIG : {seq}")
    comparison.append(f"RECON: {recon_seq}")
    comparison.append(f"ACC  : {acc:.4f}")
    comparison.append("")

    print(f"LEN={len(seq):4d} ACC={acc:.4f}", flush=True)

# ============================================================
# FINAL RESULTS
# ============================================================

final_acc = overall_correct / overall_total

print("\n==============================")
print(f"FINAL ACCURACY: {final_acc:.4f}")
print("==============================")

# ============================================================
# SAVE RESULTS
# ============================================================

with open(os.path.join(SAVE_DIR, "comparison.txt"), "w") as f:
    f.write("\n".join(comparison))

with open(os.path.join(SAVE_DIR, "local_embeddings.json"), "w") as f:
    json.dump(local_embeddings, f)

with open(os.path.join(SAVE_DIR, "global_embeddings.json"), "w") as f:
    json.dump(global_embeddings, f)

print("Saved all outputs successfully.")
