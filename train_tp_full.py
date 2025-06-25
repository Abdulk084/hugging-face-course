# tp_fsdp_train.py – Minimal end‑to‑end example: Tensor Parallel + Sequence Parallel + Loss Parallel + FSDP
"""
Launch (single node, 8 GPUs –> world_size = 8):
    torchrun --standalone --nproc_per_node 8 tp_fsdp_train.py \
        --epochs 2 --batch 8 --seq 512 --hidden 4096

Launch (multi‑node, 8 nodes × 8 GPUs = 64 ranks, 8‑way TP × 8‑way DP):
    torchrun --nnodes 8 --nproc_per_node 8 --rdzv_backend c10d \
        --rdzv_endpoint <host:port> tp_fsdp_train.py \
        --epochs 10 --batch 8 --seq 4096 --hidden 8192 --dp 8 --tp 8

The script intentionally uses random data so you can sanity‑check scaling
without a real dataset.
"""

import argparse
import math
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
    loss_parallel,
    Shard,
    Replicate,
)

# -----------------------------------------------------------------------------
#  Simple Transformer block (Llama‑style) – keeps this file self‑contained.
#  **Replace with your real model** – only the names need to match plan keys.
# -----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads  # for simplicity
        self.head_dim = hidden_dim // n_heads
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).view(B, S, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.attention = Attention(hidden_dim, n_heads)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.feed_forward = SwiGLU(hidden_dim, ffn_dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class TinyModel(nn.Module):
    """Token‑emb -> n × block -> norm -> output proj """

    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, ffn_dim):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for blk in self.layers:
            x = blk(x)
        return self.output(self.norm(x))


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def init_tp_plan(hidden_dim: int):
    """Return the dict used by parallelize_module."""
    return {
        # Attention QKV (columns) + output (rows)
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(),
        # MLP two col -> one row
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        # Sequence‑Parallel norms
        "attention_norm": SequenceParallel(),
        "ffn_norm": SequenceParallel(),
    }


def apply_tp(model: nn.Module, tp_mesh, plan):
    """Walk every TransformerBlock and the embedding/out layers."""
    # Blocks
    for blk in model.layers:
        # adjust head counts because heads are split across tp ranks
        blk.attention.n_heads //= tp_mesh.size()
        blk.attention.n_kv_heads //= tp_mesh.size()
        parallelize_module(blk, tp_mesh, plan)

    # Embedding / projector / final norm
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1), use_local_output=False
            ),
        },
    )
    return model


# -----------------------------------------------------------------------------
#  Training loop (random data)
# -----------------------------------------------------------------------------

def run(rank, args):
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group("nccl")
    world = dist.get_world_size()

    assert world % (args.dp * args.tp) == 0, "world size must equal dp*tp"
    if args.dp * args.tp != world:
        if rank == 0:
            print("[WARN] dp*tp != world. Falling back to 1‑D meshes.")
        args.dp = world // args.tp

    # ------------------------------------------------------------------ meshes
    mesh_2d = init_device_mesh(
        "cuda", (args.dp, args.tp), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]

    # ------------------------------------------------------------------ model
    model = TinyModel(
        vocab_size=args.vocab,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_dim=args.ffn,
    ).cuda()

    model = apply_tp(model, tp_mesh, init_tp_plan(args.hidden))

    model = FSDP(
        model,
        device_mesh=dp_mesh,
        use_orig_params=True,
        sharding_strategy="HYBRID_SHARD",
        mixed_precision="preferred",
    )

    # ------------------------------------------------------------------ opt
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ------------------------------------------------------------------ fake data generator
    def random_batch():
        idx = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
        tgt = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
        return idx, tgt

    # ------------------------------------------------------------------ train
    model.train()
    for epoch in range(args.epochs):
        for step in range(args.steps):
            idx, tgt = random_batch()
            optim.zero_grad(set_to_none=True)
            with loss_parallel():
                logits = model(idx)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), tgt.flatten(0, 1)
                )
                loss.backward()
            optim.step()
            if step % 10 == 0 and rank == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--ffn", type=int, default=11008)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--tp", type=int, default=8, help="tensor parallel degree")
    p.add_argument("--dp", type=int, default=1, help="data/fsdp parallel degree")
    args = p.parse_args()

    run(int(os.environ.get("LOCAL_RANK", 0)), args)
