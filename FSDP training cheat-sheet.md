Comparing with DDP, FSDP reduces GPU memory footprint by sharding model parameters, gradients, and optimizer states. It makes it feasible to train models that cannot fit on a single GPU. As shown below in the picture,

PyTorch Distributed data parallelism is a staple of scalable deep learning because of its robustness and simplicity. It however requires the model to fit on one GPU. Recent approaches like DeepSpeed ZeRO and FairScale’s Fully Sharded Data Parallel allow us to break this barrier by sharding a model’s parameters, gradients and optimizer states across data parallel workers while still maintaining the simplicity of data parallelism.

https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

I'll help you understand FSDP by building on your DDP knowledge and then show you how to convert your script. Let me break this down step by step.

## Understanding FSDP vs DDP

Think of it this way:

**DDP (your current approach):**
- Each GPU has the **complete model** (all parameters)
- During training, only gradients are shared between GPUs
- Memory usage: Full model × Number of GPUs

**FSDP (the upgrade):**
- Each GPU only stores **1/N of the model parameters** (a "shard")
- When a layer needs to run, GPUs temporarily gather all pieces
- After computation, they scatter the results back
- Memory usage: ~Full model ÷ Number of GPUs

## The Key Mental Shift

In DDP, you think: "I have the full model, let me share gradients"
In FSDP, you think: "I have 1/4 of the model, let me borrow the other 3/4 when I need them"

## Converting Your DDP Script to FSDP## Key Changes Explained

Here are the 6 main changes I made to convert your DDP script to FSDP:

### 1. **Import FSDP instead of DDP**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

### 2. **Move model to GPU BEFORE wrapping**
```python
model = AutoModelForSequenceClassification.from_pretrained(...).cuda(rank)
# THEN wrap with FSDP
```
This is crucial - FSDP expects the model to already be on the target device.

### 3. **Configure FSDP wrapping**
```python
model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,  # Wrap each transformer layer
    mixed_precision=mixed_precision_policy,         # Save even more memory
    device_id=rank,                                 # Target GPU
    sharding_strategy="FULL_SHARD"                  # Maximum sharding
)
```

### 4. **Use FSDP-specific state dict functions**
```python
with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
    model.load_state_dict(ckpt["model"])
```

### 5. **Gradient accumulation works the same**
```python
ctx = model.no_sync() if step%accum<(accum-1) else contextlib.nullcontext()
```
The API is identical to DDP!

### 6. **Save checkpoints with FSDP state dict**
```python
with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
    model_state = model.state_dict()
```

## What Happens During Training

**Forward Pass:**
1. GPU 0 has layers 1-3 of BERT
2. GPU 1 has layers 4-6 of BERT  
3. When processing layer 4, GPU 0 sends its parameters to GPU 1
4. GPU 1 computes, then sends results back

**Backward Pass:**
1. Gradients flow back through the same gather/scatter pattern
2. Each GPU only updates its own shard of parameters

## Memory Benefits You'll See

- **Before (DDP):** Each GPU holds full BERT (~440MB)
- **After (FSDP):** Each GPU holds ~220MB + temporary spikes during gather/scatter
- **With mixed precision:** Each GPU holds ~110MB

## Launch Command (Same as DDP)
```bash
torchrun --nproc_per_node 2 your_fsdp_script.py
```

The beauty of FSDP is that it maintains the same distributed training interface as DDP, but gives you massive memory savings. Your data loading, evaluation, and checkpointing logic barely changed!



Great question! Let me break down the optimizer state concept and why it's crucial for FSDP.

## What is Optimizer State?

The optimizer state is **additional data that optimizers like Adam maintain beyond just the model weights**. For Adam optimizer, this includes:

```python
# For each parameter in your model, Adam tracks:
{
    'param': actual_weight,           # The model parameter itself
    'exp_avg': momentum_buffer,       # First moment estimate (momentum)
    'exp_avg_sq': variance_buffer,    # Second moment estimate (RMSprop-like)
    'step': step_count               # How many updates for this parameter
}
```

## Memory Impact Example

Let's say you have a BERT model with 110M parameters:

**DDP approach:**
```
GPU 0: 110M weights + 110M gradients + 220M optimizer state = 440M total
GPU 1: 110M weights + 110M gradients + 220M optimizer state = 440M total
Total across 2 GPUs: 880M parameters worth of memory
```

**FSDP approach:**
```
GPU 0: 55M weights + 55M gradients + 110M optimizer state = 220M total  
GPU 1: 55M weights + 55M gradients + 110M optimizer state = 220M total
Total across 2 GPUs: 440M parameters worth of memory
```

## Why Shard Optimizer State?

**Memory reason:** Adam's optimizer state is typically **2x the size of your model weights**. For a 7B parameter model:
- Model weights: ~14GB 
- Adam optimizer state: ~28GB
- gradients ~14GB
- **Total per GPU in DDP: ~42GB +14GB**

**Consistency reason:** Each GPU only updates the parameters it owns, so it only needs the optimizer state for those parameters.

## Visual Example - What Lives Where

Imagine a simple 4-layer model on 2 GPUs:

**FSDP Sharding:**
```python
GPU 0 owns:
├── Layer 1 weights (shard 0)
├── Layer 3 weights (shard 0) 
├── Gradients for layers 1,3
└── Adam state for layers 1,3
    ├── momentum for layers 1,3
    └── variance for layers 1,3

GPU 1 owns:
├── Layer 2 weights (shard 1)
├── Layer 4 weights (shard 1)
├── Gradients for layers 2,4  
└── Adam state for layers 2,4
    ├── momentum for layers 2,4
    └── variance for layers 2,4
```

## The Communication Pattern

Here's what happens during one training step:## Why This Matters for Large Models

For really large models (like 7B+ parameters), the optimizer state becomes the **dominant memory consumer**:

```python
# 7B parameter model memory breakdown:
model_weights = 7_000_000_000 * 4  # 28GB (fp32)
adam_momentum = 7_000_000_000 * 4  # 28GB 
adam_variance = 7_000_000_000 * 4  # 28GB
gradients = 7_000_000_000 * 4      # 28GB
# Total: 112GB per GPU with DDP!

# With FSDP on 8 GPUs:
per_gpu_weights = 28GB / 8    # 3.5GB
per_gpu_adam = 56GB / 8       # 7GB  
per_gpu_gradients = 28GB / 8  # 3.5GB
# Total: 14GB per GPU - fits on consumer GPUs!
```

## The Key Insight

FSDP shards optimizer state because:

1. **Memory efficiency:** Optimizer state is huge (2x model size for Adam)
2. **Logical consistency:** Each GPU only updates parameters it owns, so it only needs optimizer state for those parameters
3. **Scalability:** Lets you train models that would never fit on a single GPU, even just for the optimizer state

This is why FSDP can train 13B+ models on consumer GPUs, while DDP would require expensive A100s with 80GB+ memory just to hold the optimizer state alone.

The gradient averaging in DDP vs the gather/scatter pattern in FSDP serves the same purpose (keeping model replicas synchronized), but FSDP does it more memory-efficiently by never requiring the full model to be resident on any single GPU.


Excellent question! You've hit on the core challenge of FSDP - **dependency management**. Let me show you exactly how this works with a concrete example.## The Key Insight About Dependencies

You're absolutely right about the dependency! Here's how FSDP handles it:

### Forward Pass - Sequential Dependency Handling
```
Layer 1 (GPU 0) → Layer 2 (GPU 1) → Layer 3 (GPU 0) → Layer 4 (GPU 1)
```

**The trick:** FSDP processes layers **in order**, one at a time. When Layer 4 needs Layer 3's output:

1. **Layer 3 computation happens first** (both GPUs participate)
2. **Layer 3's output is available** to both GPUs 
3. **Then Layer 4 computation begins** using Layer 3's output

### Backward Pass - Reverse Dependencies
```
Layer 4 ← Layer 3 ← Layer 2 ← Layer 1
```

Same sequential processing, but in reverse order.

## Communication Pattern Summary

**ALL-GATHER (Forward):**
- "I need your weights to compute this layer"
- Owner broadcasts weights to all GPUs
- Everyone computes together
- Non-owners discard the temporary copy

**REDUCE-SCATTER (Backward):**
- "We all computed gradients for this layer, let's sum them"
- All GPUs send their gradient contributions to the owner
- Owner gets the final summed gradients
- Owner uses them in optimizer step

## Memory Efficiency Magic

The beauty is that at any given moment, each GPU only holds:
- **Permanently:** Weights + optimizer state for layers it owns (~50% of model)
- **Temporarily:** Weights for the current layer being computed (~2-5% of model)

So instead of 100% model memory per GPU (DDP), you get ~52-55% peak memory per GPU (FSDP)!

This sequential processing is why FSDP has slightly more communication overhead than DDP, but the memory savings are massive for large models.

# FSDP: Sequential Layer Processing Explained

## The Key Misunderstanding

**WRONG ASSUMPTION:** "GPU 0 computes Layer 1 while GPU 1 computes Layer 2 simultaneously"

**CORRECT REALITY:** "Both GPUs work together on Layer 1, then both work together on Layer 2, etc."

---

## How FSDP Actually Works: Step-by-Step Timeline

### Setup
```
GPU 0 owns: Layer 1, Layer 3 weights (stored locally)
GPU 1 owns: Layer 2, Layer 4 weights (stored locally)

Input: batch of data = x₀
```

---

## Forward Pass: Complete Timeline

### Time Step 1: Process Layer 1 (Owner: GPU 0)

**What happens:**
1. **GPU 0 broadcasts Layer 1 weights to GPU 1**
2. **Both GPUs compute Layer 1 forward pass together**
3. **GPU 1 discards the temporary Layer 1 weights**
4. **Result: x₁ = Layer1(x₀) available on both GPUs**

```
Before:
GPU 0: [Layer1✓] [empty] [Layer3✓] [empty]  + input x₀
GPU 1: [empty] [Layer2✓] [empty] [Layer4✓]  + input x₀

During computation:
GPU 0: [Layer1✓] [empty] [Layer3✓] [empty]  + computing x₁ = Layer1(x₀)
GPU 1: [Layer1*] [Layer2✓] [empty] [Layer4✓]  + computing x₁ = Layer1(x₀)
       (*temp copy)

After:
GPU 0: [Layer1✓] [empty] [Layer3✓] [empty]  + result x₁
GPU 1: [empty] [Layer2✓] [empty] [Layer4✓]  + result x₁
```

**Key Point:** Layer 2 is NOT being computed yet!

---

### Time Step 2: Process Layer 2 (Owner: GPU 1)

**What happens:**
1. **GPU 1 broadcasts Layer 2 weights to GPU 0**
2. **Both GPUs compute Layer 2 forward pass together**
3. **GPU 0 discards the temporary Layer 2 weights**
4. **Result: x₂ = Layer2(x₁) available on both GPUs**

```
Before:
GPU 0: [Layer1✓] [empty] [Layer3✓] [empty]  + input x₁
GPU 1: [empty] [Layer2✓] [empty] [Layer4✓]  + input x₁

During computation:
GPU 0: [Layer2*] [empty] [Layer3✓] [empty]  + computing x₂ = Layer2(x₁)
GPU 1: [empty] [Layer2✓] [empty] [Layer4✓]  + computing x₂ = Layer2(x₁)
       (*temp copy)

After:
GPU 0: [Layer1✓] [empty] [Layer3✓] [empty]  + result x₂
GPU 1: [empty] [Layer2✓] [empty] [Layer4✓]  + result x₂
```

---

### Time Step 3: Process Layer 3 (Owner: GPU 0)

**What happens:**
1. **GPU 0 broadcasts Layer 3 weights to GPU 1**
2. **Both GPUs compute Layer 3 forward pass together**
3. **GPU 1 discards the temporary Layer 3 weights**
4. **Result: x₃ = Layer3(x₂) available on both GPUs**

---

### Time Step 4: Process Layer 4 (Owner: GPU 1)

**What happens:**
1. **GPU 1 broadcasts Layer 4 weights to GPU 0**
2. **Both GPUs compute Layer 4 forward pass together**
3. **GPU 0 discards the temporary Layer 4 weights**
4. **Result: logits = Layer4(x₃) available on both GPUs**

---

## The Sequential Nature: Why Dependencies Work

```
Timeline Visualization:

Time 1: [Both GPUs work on Layer 1] ────→ x₁ ready
                                          │
Time 2: [Both GPUs work on Layer 2] ────→ x₂ ready  
                                          │
Time 3: [Both GPUs work on Layer 3] ────→ x₃ ready
                                          │  
Time 4: [Both GPUs work on Layer 4] ────→ logits ready
```

**Dependencies are satisfied because:**
- Layer 2 computation starts AFTER Layer 1 is complete
- Layer 3 computation starts AFTER Layer 2 is complete
- Layer 4 computation starts AFTER Layer 3 is complete

---

## Backward Pass: Same Sequential Logic

### Time Step 1: Process Layer 4 Gradients (Owner: GPU 1)

```
Input: ∂loss/∂logits (from loss function)

Both GPUs compute:
- ∂loss/∂x₃ (to pass to Layer 3)
- ∂loss/∂Layer4_weights (for optimizer)

All GPUs send their Layer 4 gradient contributions to GPU 1
GPU 1 receives and sums all contributions
Result: GPU 1 has final Layer 4 gradients, ∂loss/∂x₃ ready for Layer 3
```

### Time Step 2: Process Layer 3 Gradients (Owner: GPU 0)

```
Input: ∂loss/∂x₃ (from Layer 4 computation)

Both GPUs compute:
- ∂loss/∂x₂ (to pass to Layer 2)  
- ∂loss/∂Layer3_weights (for optimizer)

All GPUs send their Layer 3 gradient contributions to GPU 0
GPU 0 receives and sums all contributions
Result: GPU 0 has final Layer 3 gradients, ∂loss/∂x₂ ready for Layer 2
```

### Time Step 3: Process Layer 2 Gradients (Owner: GPU 1)

### Time Step 4: Process Layer 1 Gradients (Owner: GPU 0)

---

## Why This Design Works

### 1. **No Race Conditions**
- Each layer waits for the previous layer to complete
- Sequential processing ensures data dependencies are met

### 2. **Full GPU Utilization** 
- Both GPUs participate in every layer computation
- No GPU sits idle during forward/backward pass

### 3. **Memory Efficiency**
- Each GPU permanently stores only ~50% of model weights
- Temporarily stores only current layer weights (~2-5% of model)
- Peak memory = permanent_weights + temporary_weights ≈ 52-55%

### 4. **Communication Efficiency**
- ALL-GATHER: Owner broadcasts weights to all GPUs
- REDUCE-SCATTER: All GPUs send gradients back to owner
- Alternating ownership balances communication load

---

## Common Misconception vs Reality

### ❌ **Misconception:** 
"GPU 0 computes Layer 1 and Layer 3 in parallel while GPU 1 computes Layer 2 and Layer 4 in parallel"

### ✅ **Reality:**
"All GPUs work together on Layer 1, then all work together on Layer 2, then all work together on Layer 3, then all work together on Layer 4"

**The distribution is about OWNERSHIP (where weights are stored), not COMPUTATION (who does the work).**

---

## Memory vs Computation Distribution

```
OWNERSHIP (where weights live):
GPU 0: [Layer1] [     ] [Layer3] [     ]
GPU 1: [     ] [Layer2] [     ] [Layer4]

COMPUTATION (who does the work):
Layer 1: GPU 0 + GPU 1 (both work together)
Layer 2: GPU 0 + GPU 1 (both work together)  
Layer 3: GPU 0 + GPU 1 (both work together)
Layer 4: GPU 0 + GPU 1 (both work together)
```

This is why FSDP achieves the memory benefits of model parallelism while maintaining the computational efficiency of data parallelism!

Great question! You're thinking about this logically, but there's a key insight you're missing: **Layer 1 weights are only needed ONCE during the forward pass**.

Let me clarify exactly when weights are needed:The crucial point is: **each layer's weights are only needed during that specific layer's computation**. 

Here's the detailed explanation:

## Why Weights Can Be Discarded

### Forward Pass Weight Lifecycle:
```
Layer 1 computation:
1. GPU 0 sends Layer1_weights to GPU 1
2. Both GPUs compute: x₁ = Layer1_weights × x₀  
3. Layer 1 computation is DONE
4. Layer1_weights are no longer needed for any remaining forward pass steps
5. GPU 1 safely discards Layer1_weights

Layer 2 computation:  
1. GPU 1 sends Layer2_weights to GPU 0
2. Both GPUs compute: x₂ = Layer2_weights × x₁
3. Layer 2 computation is DONE  
4. Layer2_weights are no longer needed for remaining forward pass
5. GPU 0 safely discards Layer2_weights
```

### What About Backward Pass?

The weights **will** be needed again during backward pass, but:

1. **Forward and backward are separate phases** - all forward layers complete before any backward begins
2. **When backward starts**, FSDP will do the same ALL-GATHER process again
3. **During Layer 1 backward**: GPU 0 will send Layer1_weights to GPU 1 again

## Complete Lifecycle Example:

```
FORWARD PASS:
Time 1: Layer 1 forward
  - GPU 0 → GPU 1: Send Layer1_weights  
  - Both compute Layer 1
  - GPU 1 discards Layer1_weights ✓

Time 2: Layer 2 forward  
  - GPU 1 → GPU 0: Send Layer2_weights
  - Both compute Layer 2
  - GPU 0 discards Layer2_weights ✓

Time 3: Layer 3 forward
Time 4: Layer 4 forward

BACKWARD PASS:
Time 5: Layer 4 backward
  - GPU 1 → GPU 0: Send Layer4_weights AGAIN
  - Both compute Layer 4 gradients
  - GPU 0 discards Layer4_weights ✓

Time 6: Layer 3 backward
  - GPU 0 → GPU 1: Send Layer3_weights AGAIN  
  - Both compute Layer 3 gradients
  - GPU 1 discards Layer3_weights ✓

Time 7: Layer 2 backward
Time 8: Layer 1 backward
  - GPU 0 → GPU 1: Send Layer1_weights AGAIN
  - Both compute Layer 1 gradients  
  - GPU 1 discards Layer1_weights ✓
```

## Why This Design Makes Sense:

1. **Memory Efficiency**: Keeping all temporary weights would defeat the purpose of FSDP
2. **Clear Separation**: Forward and backward are distinct phases  
3. **Predictable Pattern**: Same ALL-GATHER pattern for both forward and backward
4. **Network Optimization**: Weights are sent exactly when needed, not stored "just in case"

The key insight: **FSDP optimizes for memory usage over avoiding redundant communication**. The memory savings (storing only 50% of weights permanently) far outweigh the cost of sending weights twice (once for forward, once for backward).

Does this explain why the temporary weights are discarded?


# important for FSDP

Great question! Let me clarify how FSDP works, as there's a common misconception here.

**FSDP doesn't shard the model by giving different layers to different GPUs.** That would be pipeline parallelism. Instead, FSDP shards the **parameters within each layer** across GPUs.

## How FSDP Actually Works

In FSDP, **every GPU processes the same layers in the same order**, but each GPU only stores a fraction of each layer's parameters. Here's the key insight:

1. **Parameter Sharding**: Each layer's parameters (weights, biases) are split across all GPUs. For example, if you have a layer with 1000 parameters and 4 GPUs, each GPU stores ~250 parameters of that layer.

2. **All-Gather Before Computation**: When it's time to compute a layer, all GPUs temporarily gather the complete parameters for that layer from all other GPUs.

3. **Computation**: All GPUs run the same forward/backward computation on their respective data batches.

4. **Re-shard After Computation**: After computation, parameters are sharded again to save memory.

## Single Layer Sharding

Yes, you typically want to shard individual layers too, especially large ones. From the document:

```python
# Apply fully_shard to each layer first
for layer in model.layers:
    fully_shard(layer)
# Then apply to the root model
fully_shard(model)
```

This gives you fine-grained memory management. During forward pass of `layers[i]`, all other layers remain sharded (using minimal memory), while only `layers[i]` is temporarily unsharded for computation.

## Memory Benefits

The key advantage is that you only hold the full parameters for one layer at a time, rather than the entire model. This is why FSDP can train models that don't fit on a single GPU - you're essentially trading computation time (for all-gather operations) for memory savings.

Think of it as: instead of each GPU holding a complete copy of a smaller model (DDP), each GPU holds pieces of a much larger model (FSDP).

Excellent question! This gets to the heart of why FSDP is useful despite seeming inefficient at first glance.

## Why Use Multiple GPUs for One Layer?

**Each GPU processes a different batch of data through the same layer.** It's data parallelism, not model parallelism within the layer.

Here's what's actually happening:

```
GPU 0: processes batch_0 through layer_i (using full parameters)
GPU 1: processes batch_1 through layer_i (using full parameters)  
GPU 2: processes batch_2 through layer_i (using full parameters)
GPU 3: processes batch_3 through layer_i (using full parameters)
```

## The Key Insight

You're **parallelizing across the batch dimension**, not the computation within a single forward pass. Each GPU:

1. Gets the complete parameters for layer_i (via all-gather)
2. Processes its own unique subset of training examples
3. Computes gradients based on its data
4. Contributes its gradients to the global gradient update (via reduce-scatter)

## Why This Makes Sense

Even though there's communication overhead from all-gather, you get these benefits:

1. **Larger Effective Batch Size**: 4 GPUs = 4x the batch size you could fit on one GPU
2. **Memory Scaling**: You can train models too large for any single GPU's memory
3. **Gradient Diversity**: Each GPU contributes gradients from different data, leading to better training

## Simple Example

Imagine a layer that needs 8GB of parameters, but each GPU only has 6GB free:
- **Impossible with DDP**: Can't fit the layer on any single GPU
- **Possible with FSDP**: Each GPU stores 2GB of parameters, all-gathers to get full 8GB when needed, processes its data batch, then shards again

The communication cost is worth it because it's the only way to train the model at all!


Below I call‐out the **handful of places you must touch when you migrate a “plain” DDP training script to Fully-Sharded Data-Parallel (FSDP)** and why each change matters.
(Everything else — data loaders, loss/metrics, AMP, gradient clipping, LR scheduler, etc. — stays exactly the same.)

---

## 1  Create & initialise the process exactly as before

```python
dist.init_process_group("nccl", rank=rank, world_size=world)
torch.cuda.set_device(rank)
```

No change here.  FSDP is still a `torch.distributed` backend and inherits the same launch command (`torchrun --nproc_per_node …`). ([docs.pytorch.org][1])

---

## 2  Replace the **one-line DDP wrap** with an **FSDP wrap**

### DDP line you already have

```python
model = DistributedDataParallel(net, device_ids=[rank])
```

### FSDP (auto-wrap policy)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy

model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels).cuda(rank)

model = FSDP(
    model,
    fsdp_auto_wrap_policy=default_auto_wrap_policy,   # let FSDP decide how to split
    cpu_offload = CPUOffload(offload_params=True)     # optional, saves GPU RAM
)
```

*What changes?*

* **Sharding** – FSDP breaks every wrapped sub-module into 1/rank-sized parameter shards, so each GPU stores just its slice, not the full copy as in DDP.  Forward/backward all-reduce the shards on-the-fly, then immediately free them.  ([huggingface.co][2], [pytorch.org][3])
* **auto-wrap-policy** – `default_auto_wrap_policy` walks the module tree and wraps leaf blocks that exceed \~100 k parameters; you get decent savings without manual work.  You can swap in a size-based or custom policy later.  ([huggingface.co][2])

> **Do NOT keep the DDP line.**
> Nesting DDP **inside** FSDP defeats the memory benefit; nesting the other way around is unsupported.  Pick one or the other.  ([docs.pytorch.org][4])

---

## 3  Optimizer, scaler, scheduler: unchanged

Because FSDP exposes the same `.parameters()` iterator, construct them exactly as before:

```python
opt    = AdamW(model.parameters(), lr=base_lr*world)
scaler = GradScaler(device_type="cuda")
sched  = StepLR(opt, step_size=1, gamma=0.8)
```

No extra hooks are required. FSDP’s internal hooks run automatically after the backward pass. ([github.com][5])

---

## 4  Forward / backward loop: only one microscopic edit

```python
with model.no_sync() …                 # <-- identical helper still exists
```

`FSDP.no_sync()` skips the all-reduce of *gradients* exactly like the DDP helper (useful for gradient accumulation).  Your AMP / clipping / scaler logic stays the same. ([osc.edu][6])

---

## 5  Checkpointing: use `state_dict_type`

The big difference is **how you save / load**:

```python
from torch.distributed.fsdp import (
        StateDictType, FullStateDictConfig, ShardedStateDictConfig)

# ---------- saving  ----------
with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    ckpt = {
        "model":  model.state_dict(),     # rank-0 gets a single shard file
        "opt":    opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch":  epoch,
        "gstep":  gstep,
    }
    if rank == 0:
        torch.save(ckpt, "ckpt.pt")

# ---------- loading ----------
with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    if rank == 0:
        ckpt = torch.load("ckpt.pt", map_location="cpu")
    dist.barrier()                       # make sure file is present
    ckpt = dist.broadcast_object_list([ckpt], src=0)[0]
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
```

Why?

* A **full** state-dict would re-gather every shard on every GPU (blows memory).
* `SHARDED_STATE_DICT` lets each rank save / load only its slice; `rank0_only=True` collapses them into *one* file for convenience.  ([docs.pytorch.org][7], [discuss.pytorch.org][8])

---

## 6  What *didn’t* change?

| Part                    | Same code as DDP? | Reason                                              |
| ----------------------- | ----------------- | --------------------------------------------------- |
| Data loaders & samplers | ✅                 | FSDP is still data-parallel.                        |
| AMP / GradScaler        | ✅                 | Operates on activations, unaffected by sharding.    |
| Gradient clipping       | ✅                 | Call *after* `scaler.unscale_(opt)`.                |
| LR scheduler            | ✅                 | Works with any `torch.optim.Optimizer`.             |
| Logging / tensorboard   | ✅                 | Still use `dist.all_reduce()` for averaged metrics. |

---

## Quick mental checklist when porting DDP → FSDP

1. **Remove `DistributedDataParallel`** – replace with `FullyShardedDataParallel`.
2. **Pick a wrap policy** – start with `default_auto_wrap_policy`; tweak later.
3. **(Optional) cpu\_offload** – enables off-GPU parameter storage, slower but lighter.
4. **Use `state_dict_type` context** when you save or load.
5. Everything else can stay exactly the same.

Follow that recipe and you’ll move from a memory-hungry *replicated* model to a lean sharded version in \~15 lines of diff.

[1]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html?utm_source=chatgpt.com "Getting Started with Fully Sharded Data Parallel (FSDP2)"
[2]: https://huggingface.co/docs/accelerate/usage_guides/fsdp?utm_source=chatgpt.com "Fully Sharded Data Parallel - Hugging Face"
[3]: https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/?utm_source=chatgpt.com "Introducing PyTorch Fully Sharded Data Parallel (FSDP) API"
[4]: https://docs.pytorch.org/docs/stable/fsdp.html?utm_source=chatgpt.com "FullyShardedDataParallel — PyTorch 2.7 documentation"
[5]: https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_tutorial.rst?utm_source=chatgpt.com "tutorials/intermediate_source/FSDP_tutorial.rst at main · pytorch ..."
[6]: https://www.osc.edu/resources/getting_started/howto/howto_pytorch_fully_sharded_data_parallel_fsdp?utm_source=chatgpt.com "HOWTO: PyTorch Fully Sharded Data Parallel (FSDP)"
[7]: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html?utm_source=chatgpt.com "Getting Started with Distributed Checkpoint (DCP) - PyTorch"
[8]: https://discuss.pytorch.org/t/saving-loading-ckpt-with-multiple-fsdp-sub-process-units/217085?utm_source=chatgpt.com "Saving/Loading ckpt with multiple FSDP sub-process units"


One full replica per rank – DDP implements pure data parallelism. Every rank keeps an identical copy of all parameters, runs the forward/backward, then uses NCCL all-reduce to average gradients. Because the parameters already live on the device, the backward hooks can launch all-reduce operations immediately. 
pytorch.org

device_ids=[rank] – the constructor needs to know which of the visible CUDA devices holds that replica so it can schedule gradient synchronisation streams correctly


When you switch from DDP to FSDP, your saving/loading logic needs to change because under FSDP each rank only holds a *shard* of the parameters in GPU memory.  In order to write out a “full” checkpoint you must first gather (or “summon”) all of those shards back into a single state‐dict on one rank (or offload them to CPU) and then save that.  Likewise, when you later load, you’ll broadcast the full state‐dict back to all ranks so they re‐shard it locally.

Below is a minimal example of how to do this with the **auto-wrap** version of FSDP using PyTorch 2.1+ APIs.  I’m using the `FullStateDictConfig` and `StateDictType` helpers so that saving is efficient (offloading the gathered full state‐dict to CPU), but you can adjust them to your needs.

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from transformers import AutoModelForSequenceClassification

def ddp_setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def save_fsdp_checkpoint(fsdp_model: FSDP, path: str):
    """
    Gathers the full state dict (offloaded to CPU) and writes it to `path`.
    Only call this on rank 0.
    """
    # Configure to produce a full (unsharded) state dict, offloaded to CPU
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_sd = fsdp_model.state_dict(state_dict_type=StateDictType.FULL_STATE_DICT, 
                                    fsdp_state_dict_config=cfg)
    torch.save(full_sd, path)
    # full_sd tensors are on CPU, so no GPU memory blowup

def load_fsdp_checkpoint(fsdp_model: FSDP, path: str):
    """
    Loads a full (unsharded) checkpoint from `path` onto rank 0, then
    redistributes it to all ranks.  Call before training/resuming.
    """
    # Load only on rank 0
    if dist.get_rank() == 0:
        full_sd = torch.load(path, map_location="cpu")
    else:
        full_sd = None
    # Broadcast the full state dict (still on CPU) to all ranks
    full_sd = dist.broadcast_object_list([full_sd], src=0)[0]
    # Now every rank has the full state dict in CPU memory; load and re‐shard
    fsdp_model.load_state_dict(full_sd)

def main(rank: int, world_size: int, model_name: str, ckpt_path: str | None):
    ddp_setup(rank, world_size)

    # 1) build your HF model *on CPU* for FSDP to shard it immediately
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                num_labels=4)
    # 2) wrap in FSDP (no need for DDP)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=default_auto_wrap_policy,
        device_id=rank,  # puts local shards on the right GPU
    )

    # 3) optionally resume
    if ckpt_path:
        load_fsdp_checkpoint(fsdp_model, ckpt_path)

    # ... your training loop ...
    for epoch in range(5):
        # train, validate, etc.
        pass

    # 4) at the end (or every N steps) save
    if rank == 0:
        save_fsdp_checkpoint(fsdp_model, "fsdp_model_full.pt")

    dist.barrier()
    ddp_cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size, model_name="bert-base-uncased", ckpt_path=None)
```

---

## What Changed vs. DDP?

1. **Wrapping**

   ```python
   # DDP:
   model = DistributedDataParallel(model.cuda(rank), device_ids=[rank])
   ```

   vs.

   ```python
   # FSDP:
   model = AutoModelForSequenceClassification.from_pretrained(...).cpu()
   fsdp_model = FSDP(model,
                     auto_wrap_policy=default_auto_wrap_policy,
                     device_id=rank)
   ```

   You do **not** wrap in DDP first; FSDP replaces DDP entirely.

2. **Saving**

   * **DDP**: you’d do

     ```python
     if rank==0:
         torch.save(model.module.state_dict(), "ckpt.pt")
     ```
   * **FSDP**: you must gather (“unshard”) before saving:

     ```python
     full_sd = fsdp_model.state_dict(
         state_dict_type=StateDictType.FULL_STATE_DICT,
         fsdp_state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
     )
     torch.save(full_sd, "fsdp_model_full.pt")
     ```

3. **Loading / Resuming**

   * **DDP**:

     ```python
     ckpt = torch.load(path)
     model.module.load_state_dict(ckpt)
     ```
   * **FSDP**: load the full dict on rank 0, broadcast to all ranks, then `fsdp_model.load_state_dict(...)`, which re‐shards it for you.

---

### Why this matters

* Under FSDP each GPU **only** holds a shard of the parameters, gradients, and optimizer state.  If you simply called `model.state_dict()` it would only return that local shard—not the full weights!  By using `StateDictType.FULL_STATE_DICT` (and `summon_full_params` in older versions) you temporarily reconstruct the complete model weights in CPU memory so you can write them out as a single checkpoint.
* On load, FSDP will automatically split the full weight tensor back into shards appropriate for each rank.

With that pattern in place, you get both the memory‐savings of FSDP during training *and* a portable, single‐file checkpoint you can load (and even use later under DDP or single‐GPU) without extra hassle.


Yes—when you switch from DDP to FSDP, your entire training loop, optimizer/AMP setup, gradient accumulation, logging, etc. stays exactly the same. The only bits you need to add (or replace) are around saving and loading the FSDP‐wrapped model’s sharded state.

Concretely, in your “resume from checkpoint” block you:

1. **Load optimizer, scaler, and counters** on **rank 0** and broadcast those counters to all ranks (exactly as before).
2. **Pull in the full, un-sharded model weights** via `model.state_dict(state_dict_type=FULL_STATE_DICT, fsdp_state_dict_config=...)` and then call `model.load_state_dict(...)`.

Everything else—your forward/backward, `scaler.step()`, `scaler.update()`, `optimizer.step()`, `dist.all_reduce()` logging, gradient accumulation, etc.—remains untouched.

---

### Minimal “resume” snippet under FSDP

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# … after ddp_setup(), model = FSDP(...), opt, scaler, etc. …

global_step, start_epoch = 0, 0

if resume_path:
    # 1️⃣ Load opt+scaler+step counters on rank 0
    ckpt = torch.load(resume_path, map_location="cpu")
    opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
    global_step  = ckpt["global_step"] + 1
    start_epoch  = ckpt["epoch"] + 1
    if rank == 0:
        print(f"[rank0] resumed from {resume_path} "
              f"(epoch={start_epoch}, gstep={global_step})")

# 2️⃣ Broadcast counters so all ranks agree
ep_t   = torch.tensor([start_epoch], device=rank)
gs_t   = torch.tensor([global_step], device=rank)
dist.broadcast(ep_t, src=0)
dist.broadcast(gs_t, src=0)
start_epoch = int(ep_t.item())
global_step = int(gs_t.item())

# 3️⃣ Load the full model state and re-shard on each GPU
if resume_path:
    fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_sd = model.state_dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        fsdp_state_dict_config=fsdp_cfg
    )
    model.load_state_dict(full_sd)
```

* **Step 1** is identical to your DDP code, except you skip loading `"model"` here—FSDP handles that in step 3.
* **Step 2** is also the same broadcast you already have.
* **Step 3** is the only FSDP-specific bit: you pull in a *full* (un-sharded) `state_dict` on rank 0 (and, behind the scenes, each rank participates in the gather) then hand that to your `model.load_state_dict()`, which re-shards it for local use.

After that, your training loop runs exactly as before—no other changes required.


# FSDP Interview Guide - Key Concepts & Code Examples

## 1. Core FSDP Concepts

### What is FSDP?
- **Fully Sharded Data Parallel** - shards model parameters, gradients, and optimizer states across GPUs
- **Memory Efficiency**: Enables training larger models by reducing per-GPU memory usage
- **vs DDP**: DDP replicates the full model on each GPU; FSDP shards it

### Key Differences from DDP
```python
# DDP - Full model on each GPU
model = DDP(model, device_ids=[rank])
# Each GPU: Full model parameters + gradients

# FSDP - Sharded model across GPUs
model = FSDP(model, device_id=rank)
# Each GPU: Only subset of parameters + gradients
```

## 2. FSDP Wrapping Strategies

### Auto Wrap Policy
```python
from torch.distributed.fsdp.wrap import default_auto_wrap_policy, transformer_auto_wrap_policy

# Default - wraps by parameter count
model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)

# Transformer-specific - wraps transformer blocks
from transformers.models.bert.modeling_bert import BertLayer
model = FSDP(model, 
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={BertLayer}
    )
)
```

### Manual Wrapping
```python
# Wrap specific modules manually
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FSDP(BertModel(...))  # Wrap encoder
        self.classifier = nn.Linear(768, num_classes)  # Don't wrap classifier

# Or wrap the entire model
model = FSDP(MyModel(), auto_wrap_policy=my_policy)
```

## 3. CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

# Offload parameters to CPU when not in use
model = FSDP(model, 
    cpu_offload=CPUOffload(offload_params=True),
    device_id=rank
)

# Advanced offloading
model = FSDP(model,
    cpu_offload=CPUOffload(
        offload_params=True,
        offload_gradients=True  # Also offload gradients
    )
)
```

## 4. State Dict Management

### The Three State Dict Types
```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# 1. LOCAL_STATE_DICT - Each rank saves its shard only
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    local_sd = model.state_dict()

# 2. SHARDED_STATE_DICT - Distributed saving/loading
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sharded_sd = model.state_dict()

# 3. FULL_STATE_DICT - Complete model on rank 0 (most common)
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    full_sd = model.state_dict()  # Only rank 0 gets full dict
```

### Checkpoint Save/Load Pattern
```python
def save_checkpoint(model, optimizer, path, rank):
    if rank == 0:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        torch.save({"model": state_dict, "optimizer": optimizer.state_dict()}, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
```

## 5. Gradient Synchronization Control

```python
# Gradient accumulation with FSDP
for step, batch in enumerate(dataloader):
    # Skip sync on intermediate steps
    if step % accumulation_steps < (accumulation_steps - 1):
        with model.no_sync():
            loss = model(batch).loss / accumulation_steps
            loss.backward()
    else:
        # Sync on final step
        loss = model(batch).loss / accumulation_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 6. Mixed Precision with FSDP

```python
from torch.distributed.fsdp import MixedPrecision
import torch

# Configure mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.float16,      # Parameters in fp16
    reduce_dtype=torch.float16,     # Gradients reduced in fp16
    buffer_dtype=torch.float16,     # Buffers in fp16
)

model = FSDP(model, 
    mixed_precision=mp_policy,
    device_id=rank
)

# Use with autocast
with torch.autocast("cuda"):
    output = model(inputs)
    loss = criterion(output, targets)
```

## 7. Common FSDP Patterns & Best Practices

### Initialization Order
```python
# CORRECT: Model on CPU first, then wrap with FSDP
model = MyModel()  # Model created on CPU
model = FSDP(model, device_id=rank)  # FSDP moves to GPU

# WRONG: Don't move to GPU before FSDP
# model = MyModel().cuda()  # Don't do this
# model = FSDP(model)
```

### Optimizer Creation
```python
# Create optimizer AFTER FSDP wrapping
model = FSDP(model, device_id=rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Scale learning rate by world size for distributed training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * world_size)
```

### Memory Management
```python
# Use set_to_none=True for better memory efficiency
optimizer.zero_grad(set_to_none=True)

# Flatten parameters for memory efficiency
model = FSDP(model, 
    flatten_parameters=True,  # Default is True
    device_id=rank
)
```

## 8. Debugging & Monitoring

### Memory Usage
```python
def print_memory_stats(rank):
    if rank == 0:
        print(f"GPU {rank} Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"GPU {rank} Max Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

# Call after each major operation
print_memory_stats(rank)
```

### Parameter Sharding Info
```python
# Check which parameters are sharded
for name, param in model.named_parameters():
    if hasattr(param, '_sharded'):
        print(f"{name}: sharded={param._sharded}")
```

## 9. Common Interview Questions & Answers

### Q: When would you use FSDP over DDP?
**A**: When model size exceeds single GPU memory, need to train larger models, or want better memory efficiency. FSDP shards parameters; DDP replicates them.

### Q: What's the main challenge with FSDP checkpointing?
**A**: Managing different state dict types. Must use `FULL_STATE_DICT` with proper config for standard checkpointing, or implement sharded checkpointing for very large models.

### Q: How does gradient accumulation work differently in FSDP?
**A**: Must use `model.no_sync()` context manager to prevent gradient synchronization on intermediate steps, only sync on the final accumulation step.

### Q: What happens during forward/backward pass in FSDP?
**A**: 
- **Forward**: All-gather parameters just before computation, discard after
- **Backward**: All-gather parameters for gradient computation, reduce-scatter gradients, discard parameters

### Q: How do you handle model initialization with FSDP?
**A**: Always create model on CPU first, then wrap with FSDP. FSDP will handle moving shards to appropriate devices.

## 10. Performance Optimization Tips

```python
# 1. Use transformer-specific auto-wrap for transformers
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={TransformerBlock}
)

# 2. Enable CPU offloading for large models
cpu_offload = CPUOffload(offload_params=True)

# 3. Use mixed precision
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
)

# 4. Optimize data loading
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    pin_memory=True,        # Faster CPU->GPU transfer
    num_workers=4,          # Parallel data loading
    persistent_workers=True # Keep workers alive
)

# 5. Use compile for PyTorch 2.0+
model = torch.compile(model)  # After FSDP wrapping
```

## Key Takeaway for Interviews
FSDP is about **memory efficiency through sharding** - understand the trade-offs between memory usage, communication overhead, and complexity compared to DDP. Focus on practical implementation details like state dict management and gradient synchronization control.



# Code Comparison:
# Pure PyTorch (verbose but explicit):
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # validation logic...
    
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_{epoch}.pt')
    # Handle early stopping, logging, etc.

```

# PyTorch Lightning (concise, handles boilerplate):
```python
class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        output = self(batch['input'])
        loss = self.criterion(output, batch['target'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        # validation logic...
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Everything else (checkpointing, logging, etc.) handled automatically
trainer = pl.Trainer(gpus=4, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```



Below is a breakdown of the **model initialization** contract for FSDP-2’s `fully_shard` API, with an example showing how you would apply it “bottom-up” on your submodules before the root model. This is the first, crucial step you must take before any forward or backward passes.

---

## 1. What `fully_shard(model)` Does at Initialization

When you call `fully_shard(model)` on an `nn.Module`, FSDP-2 replaces each parameter tensor in place with a **DTensor**, sharding it (by default) along its dim-0 across the data-parallel workers:

1. **In-place parameter conversion (to DTensor):**
   All of `model.parameters()` are converted from plain `torch.Tensor` to `DTensor`. Each DTensor is split (e.g. via `torch.chunk(dim=0)`) across the GPUs in the process group.
   ([docs.pytorch.org][1])

Original W:
 tensor([[1., 2., 3., 4.],
         [5., 6., 7., 8.]])

Rank 0 local shard:
 tensor([[1., 2., 3., 4.]])

Rank 1 local shard:
 tensor([[5., 6., 7., 8.]])

Reconstructed full W:
 tensor([[1., 2., 3., 4.],
         [5., 6., 7., 8.]])

2. **Device placement:**
   The shards are immediately moved to the correct device based on your device mesh (usually `cuda:rank`).
   ([docs.pytorch.org][1])

3. **Type “unioning”:**
   Internally, the `type(model)` is modified *in place* so that your original module now is also an `FSDPModule` (e.g. `FSDPLinear` for `nn.Linear`). You get both the original API and FSDP hooks (e.g. `reshard()`, `unshard()`).
   ([docs.pytorch.org][1])

---

## 2. Why You Should Wrap Submodules First (“Bottom-Up”)

FSDP-2 groups parameters into all-gather groups at each wrapping level to minimize communication:

* **Per-layer wrapping:**
  For something like a Transformer, you should call `fully_shard(layer)` on *each* layer first. That ensures each layer’s parameters get their own DTensor and are sharded locally.
* **Root model wrapping:**
  After all the large submodules are sharded, you call `fully_shard(model)` on the *entire* model. FSDP will then exclude parameters already wrapped in each layer and group the *remaining* (e.g. embeddings, final classifier head) for its own all-gather.

This bottom-up approach guarantees that FSDP doesn’t accidentally lump many unrelated parameters into one huge communication group, and lets it optimize each all-gather for the right granularity.
([docs.pytorch.org][1])

---

## 3. Triggering Hooks on Forward/Backward

* **Pre-forward/backward hooks** automatically all-gather the sharded DTensors back into full `torch.Tensor` parameters just in time for your `model(input)` call.
* **Post-forward/backward hooks** then free the unsharded tensors and revert parameters to the DTensor format, so that memory usage stays low until the next step.

You do **not** call `model.forward()` directly, but always `model(input)` so that the hooks fire correctly.
([docs.pytorch.org][1])

---

## 4. Example

```python
from torch.distributed.fsdp import fully_shard, FSDPModule

# 1) Build your model
model = MyTransformerModel(...)   # e.g., has model.layers = ModuleList([...])

# 2) Shard each transformer layer first
for layer in model.layers:
    fully_shard(layer)

# 3) Shard the remaining parts (embeddings, output head, etc.)
fully_shard(model)

# Now `model` is an FSDP-wrapped module:
assert isinstance(model, FSDPModule)
```

* **After step 2**, each `layer`’s parameters are independently sharded (their DTensors know exactly how to all-gather shards).
* **After step 3**, the parameters *not* in those layers (e.g. the embedding table, the final linear head) are sharded as a separate group.

---

## 5. Key Benefits over DDP and FSDP-1

* **Per-parameter sharding** rather than flatten-and-shard: easier to reason about and adaptive to frozen parameters.
* **DTensor-based**: you get sharded parameters that carry metadata about the device mesh, making communication and checkpointing more flexible.
* **No extra wrapper modules**: `fully_shard` adds hooks to your existing modules instead of creating wrapper classes around them.
  ([docs.pytorch.org][1])

---

**In summary**, `fully_shard` at init:

* Converts each parameter to a DTensor and shards it (dim-0) across GPUs
* Modifies the module’s type to include FSDP hooks
* Requires you to wrap *submodules first*, then the root model, to get optimal sharding groups
* Sets the stage for all-gather and free-and-reshard hooks on forward/backward

Once you’ve called `fully_shard` bottom-up, you can proceed with your usual training loop: forward, loss, backward, and optimizer step—all the DTensor sharding and communication is handled transparently behind the scenes.

[1]: https://docs.pytorch.org/docs/main/_sources/distributed.fsdp.fully_shard.md.txt "docs.pytorch.org"


You haven’t “gone wrong” — you’re simply using the **auto-wrap** API instead of the low-level `fully_shard(...)` calls. Here’s why that’s fine, and how the two approaches map to one another:

---

## 1. `fully_shard(module)` vs. `FSDP(..., auto_wrap_policy=...)`

| Low-level API (`fully_shard`)                                                                    | High-level API (`FSDP` w/ auto-wrap)                                                                                                                                           |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| You manually call `fully_shard(submodule)` on each layer, then `fully_shard(model)` on the root. | You wrap the **entire** `model` once in `FSDP(...)` and pass in an `auto_wrap_policy`.                                                                                         |
| Gives you explicit, per-layer sharding control.                                                  | FSDP inspects your model’s structure and automatically wraps (i.e. shards) any `nn.Module` whose parameter count exceeds a threshold, grouping submodules “bottom-up” for you. |

* **With `fully_shard`** you write:

  ```python
  for layer in model.layers:
      fully_shard(layer)
  fully_shard(model)
  ```
* **With auto-wrap** you write:

  ```python
  model = FSDP(
      model,
      fsdp_auto_wrap_policy=default_auto_wrap_policy,
      cpu_offload=CPUOffload(...),
      device_id=rank,
  )
  ```

  Under the hood, `default_auto_wrap_policy` will dive into your `model`, find large submodules, and call the equivalent of `fully_shard(...)` on each before finally wrapping any leftover parameters at the root.

---

## 2. Why you don’t need both

* **Don’t mix**: if you manually `fully_shard` submodules *and* then wrap the root with auto-wrap, you’ll end up double-wrapping (and FSDP will complain or do redundant work).
* **Pick one**:

  * **Manual sharding** (`fully_shard`) when you need absolute control over exactly which layers are sharded.
  * **Auto-wrap** when you want FSDP to shard everything that “looks big” for you.

You chose **auto-wrap**, so you **shouldn’t** also call `fully_shard(...)`.

---

## 3. When to use each in practice

* **Manual**

  ```python
  from torch.distributed.fsdp import fully_shard
  for layer in model.layers:
      fully_shard(layer)
  fully_shard(model)
  ```

  Use if you know exactly which custom submodules you want sharded (e.g. non-standard blocks).

* **Auto-wrap**

  ```python
  model = FSDP(
      model,
      fsdp_auto_wrap_policy=default_auto_wrap_policy,
      …
  )
  ```

  Use for standard PyTorch modules (Transformers, ResNets, etc.). It “just works” without boilerplate.

---

### TL;DR

> You **don’t** need to sprinkle `fully_shard(...)` calls if you’re already wrapping your `model` in
>
> ```python
> FSDP(model, fsdp_auto_wrap_policy=default_auto_wrap_policy, …)
> ```
>
> That single line will internally shard each submodule (bottom-up) and then the root, exactly as if you’d called `fully_shard` yourself.

Here’s what’s going on in that snippet, step by step:

1. **Parameters become DTensors**
   When you call `fully_shard(model)`, FSDP replaces each `torch.Tensor` parameter in your `nn.Module` with a `DTensor` that represents just its local shard.  Each shard is an actual `torch.Tensor` holding only that rank’s piece of the parameter—by default FSDP shards along dimension 0 (rows).

2. **Checking the type**

   ```python
   from torch.distributed._tensor import DTensor
   from torch.distributed._tensor.placement_specs import Shard

   for param in model.parameters():
       assert isinstance(param, DTensor)
       assert param.placements == (Shard(dim=0),)
   ```

   * `isinstance(param, DTensor)` confirms FSDP has wrapped it.
   * `param.placements == (Shard(0),)` shows it’s sharded across dim 0.

3. **Inspecting your local shard**
   You can peek at what each rank actually holds via:

   ```python
   local_tensor = param.to_local()
   ```

   That returns the underlying `torch.Tensor` for this rank (e.g. rank 0 holds rows 0…N⁄R−1, rank 1 holds rows N⁄R…2N⁄R−1, etc).

4. **Optimizer sees DTensors just fine**
   You build your optimizer *after* wrapping in FSDP:

   ```python
   optim = torch.optim.Adam(model.parameters(), lr=1e-2)
   ```

   Under the hood, gradient all-reduces and parameter updates all use the DTensor abstraction. You don’t need a special “FSDP optimizer.”

5. **Grad-clipping works out of the box**
   Whether you call

   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

   on a DDP-wrapped model or an FSDP-wrapped one, it will unshard (all-gather) any needed gradient pieces, clip globally, then scatter the clipped results back down to each shard.

6. **Checkpointing via state dicts**

   * **Sharded checkpoints**:  `model.state_dict()` will by default give you each rank’s *local* shards.
   * **Full (unsharded) checkpoints**:  use the `with FSDP.state_dict_type(…)` context (as in the earlier snippet) to gather the full, rank-0-only state dict on CPU.

---

### Why DTensor helps

* **Memory savings**:  each GPU holds only 1⁄R of each parameter, gradient, and optimizer state.
* **Seamless API**:  you write `model.parameters()`, `optim.step()`, and `clip_grad_norm_()` exactly as usual; DTensor makes it “just work.”
* **Flexibility**:  you can change shard dimension, CPU-offload, activation checkpointing, etc., entirely through FSDP flags—without rewriting optimizer or training logic.


Below is a concise, example-driven explanation of how FSDP overlaps (“prefetches”) parameter all-gathers with computation in the forward and backward passes, and what “implicit” vs “explicit” prefetching means.

---

## 1. How FSDP hooks all-gather and re-shard parameters

When you call `FullyShardedDataParallel(model)`, under the hood FSDP:

1. **Replaces each `nn.Parameter` with a sharded `DTensor`** that only holds a slice of the full tensor on each rank.
2. **Registers forward/backward hooks** on the *flat parameter* buffer, so that just before you need the full parameter you automatically do an **all-gather** across ranks, and just after you free or re-shard it again.

This means you never have to write explicit communication calls—FSDP transparently gathers the shards when you enter a layer’s `forward`, and then shards them back after you leave that layer.

---

## 2. Implicit vs Explicit Forward Prefetching

### 2.1 Implicit prefetching (the default)

* FSDP uses a **separate NCCL stream** internally for all-gathers.
* When you call `loss = model(x)`, each layer’s forward does:

  1. **All-gather** its parameters into the all-gather buffer (on the secondary stream).
  2. **Compute** its forward pass using those parameters (on the default stream).
* Because the all-gather is on a different CUDA stream, the **all-gather for layer *n* can overlap** with the compute of layer *n−1*, even though the CPU enqueued them in sequence ([docs.pytorch.org][1]).

### 2.2 Explicit forward prefetching

* You can opt into **`forward_prefetch=True`** to have FSDP issue the *next* layer’s all-gather **ahead of** the current layer’s compute.
* That is, for a static-graph model with a fixed layer order, FSDP will do:

  ```
  allgather(layer0)
  allgather(layer1)      ← prefetch
  compute(layer0)
  compute(layer1)
  ```
* This can eke out a bit more overlap, but *only* works reliably when the layer execution order never changes between iterations (e.g. simple Transformers) ([docs.pytorch.org][1]).
* Most real-world LLM workloads find the **implicit** scheme sufficient, so explicit prefetching is rarely needed.

---

## 3. Why explicit backward prefetching is required

Backward communication uses the *same* NCCL process group (hence the same CUDA stream) for both all-gathers and reduce-scatters.  By default, every kernel enqueue blocks further communication on that stream, so without reordering:

```
… backward of layer N …
reduce_scatter(parameters N gradients)
all_gather(parameters N−1)   ← this must wait until reduce_scatter is done
backward(layer N−1)
…
```

You end up with **zero overlap** between gradient communication for layer *N* and compute for layer *N−1*.

FSDP therefore **enables explicit backward prefetching** by default, which reorders the CPU enqueues so that you do:

```
all_gather(params N−1)      ← prefetch next layer’s parameters
reduce_scatter(grads N)     ← current layer’s gradients
backward(layer N−1)
```

This reordering allows the next layer’s all-gather to overlap with the current layer’s reduce-scatter ([docs.pytorch.org][1]).

---

## 4. Putting it all together: what you see in your loop

With FSDP set up, your training loop doesn’t change—just write:

```python
for xb, yb in train_loader:
    optimizer.zero_grad()
    loss = model(xb).loss
    loss.backward()
    optimizer.step()
```

Under the covers, FSDP:

* **Shards** parameters and optimizer state across ranks.
* **All-gathers** each layer’s parameters just before use, on a separate CUDA stream (implicit forward prefetch).
* **Re-shards** them immediately after use.
* During backward, **prefetches** the *next* layer’s parameters before doing the current layer’s gradient reduce-scatter, thus overlapping communication and compute.

---

### References

* PyTorch FSDP “Prefetch Nuances” documentation ([docs.pytorch.org][1])
* PyTorch FSDP blog introducing the API ([docs.pytorch.org][2])
* PyTorch distributed overview of all-gather/reduce-scatter under FSDP ([docs.pytorch.org][1])

[1]: https://docs.pytorch.org/docs/stable/notes/fsdp.html "FSDP Notes — PyTorch 2.7 documentation"
[2]: https://docs.pytorch.org/docs/stable/notes/fsdp.html?utm_source=chatgpt.com "FSDP Notes — PyTorch 2.7 documentation"

Here’s a concise overview and deep dive into the Advanced FSDP tutorial, covering its core ideas and practical patterns.

At a high level, Fully Sharded Data Parallel (FSDP) shards model parameters, gradients, and optimizer state across GPUs to drastically reduce per-card memory overhead compared to DDP. During the forward pass, each FSDP unit all-gathers the shards it needs, runs its computation, then discards non-owned shards; the backward pass similarly all-gathers before computing gradients and then uses reduce-scatter to synchronize them. This tutorial walks through six key FSDP features—auto-wrapping policies for Transformers, mixed precision, device-aware initialization, sharding strategies (Zero-2 vs. Zero-3), backward prefetching, and efficient checkpointing by streaming full state dicts to CPU on rank 0—illustrated via fine-tuning a T5 model on WikiHow. Practical code patterns show how to wrap submodules, control gradient synchronization with `no_sync()`, clip gradients, schedule learning rates, and save/load sharded checkpoints in distributed settings. The result is a scalable blueprint for training billion-parameter models on multi-GPU setups without running out of memory.

## 1. FSDP Fundamentals

### 1.1 Sharding Parameters, Gradients, and Optimizer State

* **Constructor**: FSDP shards each module’s parameters along dimension-0 so each rank holds only its slice ([docs.pytorch.org][1]).
* **Forward Pass**: For each wrapped submodule, FSDP all-gathers full parameters, runs the forward, then drops non-owned shards to free memory ([docs.pytorch.org][1]).
* **Backward Pass**: It all-gathers again to compute gradients, discards non-owned shards, then uses `reduce_scatter` to shard the gradient update ([docs.pytorch.org][1]).

### 1.2 Overlap of Communication and Computation

Implicit prefetching overlaps the all-gather of the next FSDP unit with current computation by queuing all-gathers on a separate CUDA stream, improving throughput without extra code ([docs.pytorch.org][1]).

## 2. Key Features in the Tutorial

### 2.1 Transformer Auto-Wrap Policy

Automatically wraps large layers (e.g., `T5Block`) into FSDP units to balance shard granularity and communication cost.

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})
model = FSDP(model, auto_wrap_policy=auto_wrap, …)
```

This ensures embedding tables and shared modules are wrapped at the correct hierarchy for efficiency ([docs.pytorch.org][1]).

### 2.2 Mixed Precision

FSDP’s `MixedPrecision` lets you shard parameters, communication buffers, and gradients in lower precision (e.g., bfloat16) while keeping computation in float32 if desired:

```python
from torch.distributed.fsdp import MixedPrecision
mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
model = FSDP(model, mixed_precision=mp, …)
```

This can yield up to 4× speedups and \~30% memory savings on Ampere GPUs ([docs.pytorch.org][1]).

### 2.3 Initializing on Device

By passing `device_id=torch.cuda.current_device()`, FSDP moves each unit to its GPU as it’s wrapped, avoiding a full upfront CPU→GPU copy that might OOM for very large models ([docs.pytorch.org][1]).

### 2.4 Sharding Strategy (Zero-2 vs. Zero-3)

* **FULL\_SHARD (Zero-3)**: Shards parameters, gradients, and optimizer state—max memory savings, more communication.
* **SHARD\_GRAD\_OP (Zero-2)**: Only shards gradients and optimizer state—fewer all-gathers in backward, slightly higher memory ([docs.pytorch.org][1]).

```python
from torch.distributed.fsdp import ShardingStrategy
model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, …)
```

### 2.5 Backward Prefetching

Controls when to request the next unit’s parameters during backprop:

* `BackwardPrefetch.BACKWARD_PRE` overlaps comm with computation for speed
* `BackwardPrefetch.BACKWARD_POST` waits to minimize memory ([docs.pytorch.org][1]).

### 2.6 Checkpointing by Streaming to CPU

Efficiently save large models by offloading shards to CPU on rank 0 one at a time:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state = model.state_dict()
if rank == 0:
    torch.save(state, "model.pt")
```

This avoids GPU OOM during `state_dict()` and makes loading straightforward ([docs.pytorch.org][1]).

## 3. Putting It All Together

In practice, you:

1. **Setup DDP**: Initialize NCCL, set device per rank.
2. **Build DataLoaders** with `DistributedSampler`.
3. **Instantiate Model** on CPU, then wrap with FSDP auto-wrap, mixed precision, CPU offload, and device\_id.
4. **Optimizer & Scheduler**: scale learning rate by world\_size, step after `optimizer.step()`.
5. **Training Loop**:

   * Use `model.no_sync()` for gradient accumulation blocks.
   * Wrap forward/backward in `autocast`.
   * Clip grads, unscale, `scaler.step()`, `scaler.update()`.
   * Periodically `dist.all_reduce()` for logging.
   * Save sharded checkpoints on rank 0 via streaming API.
6. **Evaluation**: switch modes, all\_reduce stats, switch back to train.

This flow lets you seamlessly train T5-3B (or larger) on 8 × A100 without running out of GPU memory, while still leveraging full data parallel scaling and mixed precision.

[1]: https://docs.pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html "Advanced Model Training with Fully Sharded Data Parallel (FSDP) — PyTorch Tutorials 2.5.0+cu124 documentation"


Advanced Model Training with Fully Sharded Data Parallel (FSDP)
===============================================================

**Author**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Less
Wright <https://github.com/lessw2020>`__, `Rohan Varma
<https://github.com/rohan-varma/>`__, `Yanli Zhao
<https://github.com/zhaojuanmao>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
      :class-card: card-prerequisites

      * PyTorch's Fully Sharded Data Parallel Module: A wrapper for sharding module parameters across
      data parallel workers.




   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
      :class-card: card-prerequisites

      * PyTorch 1.12 or later
      * Read about the `FSDP API <https://pytorch.org/docs/main/fsdp.html>`__.


This tutorial introduces more advanced features of Fully Sharded Data Parallel
(FSDP) as part of the PyTorch 1.12 release. To get familiar with FSDP, please
refer to the `FSDP getting started tutorial
<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__.

In this tutorial, we fine-tune a HuggingFace (HF) T5 model with FSDP for text
summarization as a working example.

The example uses Wikihow and for simplicity, we will showcase the training on a
single node, P4dn instance with 8 A100 GPUs. We now have several blog posts (
`(link1), <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__
`(link2) <https://engineering.fb.com/2021/07/15/open-source/fsdp/>`__)
and a `paper <https://arxiv.org/abs/2304.11277>`__ on
large scale FSDP training on a multi-node cluster.

FSDP is a production ready package with focus on ease of use, performance, and
long-term support.  One of the main benefits of FSDP is reducing the memory
footprint on each GPU. This enables training of larger models with lower total
memory vs DDP, and leverages the overlap of computation and communication to
train models efficiently.
This reduced memory pressure can be leveraged to either train larger models or
increase batch size, potentially helping overall training throughput.  You can
read more about PyTorch FSDP `here
<https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__.


FSDP Features in This Tutorial
------------------------------
* Transformer Auto Wrap Policy
* Mixed Precision
* Initializing FSDP Model on Device
* Sharding Strategy
* Backward Prefetch
* Model Checkpoint Saving via Streaming to CPU



Recap on How FSDP Works
-----------------------

At a high level FDSP works as follow:

*In the constructor*

* Shard model parameters and each rank only keeps its own shard

*In the forward pass*

* Run `all_gather` to collect all shards from all ranks to recover the full
  parameter for this FSDP unit and run the forward computation
* Discard the non-owned parameter shards it has just collected to free memory

*In the backward pass*

* Run `all_gather` to collect all shards from all ranks to recover the full
  parameter in this FSDP unit and run backward computation
* Discard non-owned parameters to free memory.
* Run reduce_scatter to sync gradients


Fine-tuning HF T5
-----------------
HF T5 pre-trained models are available in four different sizes, ranging from
small with 60 Million parameters to XXL with 11 Billion parameters. In this
tutorial, we demonstrate the fine-tuning of a T5 3B with FSDP for text
summarization using WikiHow dataset.  The main focus of this tutorial is to
highlight different available features in FSDP that are helpful for training
large scale model above 3B parameters. Also, we cover specific features for
Transformer based models. The code for this tutorial is available in  `Pytorch
examples
<https://github.com/pytorch/examples/tree/main/distributed/FSDP/>`__.


*Setup*

1.1 Install the latest PyTorch

.. code-block:: bash

    pip3 install torch torchvision torchaudio

1.2 Dataset Setup

Please create a `data` folder, download the WikiHow dataset from `wikihowAll.csv
<https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358>`__  and
`wikihowSep.cs <https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag>`__,
and place them in the `data` folder.  We will use the wikihow dataset from
`summarization_dataset
<https://github.com/pytorch/examples/blob/main/distributed/FSDP/summarization_dataset.py>`__.

Next, we add the following code snippets to a Python script “T5_training.py”.

.. note::
   The full source code for this tutorial is available in `PyTorch examples
   <https://github.com/pytorch/examples/tree/main/distributed/FSDP/>`__.

1.3  Import necessary packages:

.. code-block:: python

    import os
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from transformers import AutoTokenizer, GPT2TokenizerFast
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import functools
    from torch.optim.lr_scheduler import StepLR
    import torch.nn.functional as F
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from transformers.models.t5.modeling_t5 import T5Block

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
     checkpoint_wrapper,
     CheckpointImpl,
     apply_activation_checkpointing_wrapper)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    from functools import partial
    from torch.utils.data import DataLoader
    from pathlib import Path
    from summarization_dataset import *
    from transformers.models.t5.modeling_t5 import T5Block
    from typing import Type
    import time
    import tqdm
    from datetime import datetime

1.4 Distributed training setup.
Here we use two helper functions to initialize the processes for distributed
training,  and then to clean up after training completion.  In this tutorial, we
are going to use torch elastic, using `torchrun
<https://pytorch.org/docs/stable/elastic/run.html>`__ , which will set the
worker `RANK` and `WORLD_SIZE` automatically.

.. code-block:: python

    def setup():
        # initialize the process group
        dist.init_process_group("nccl")

    def cleanup():
        dist.destroy_process_group()

2.1  Set up the HuggingFace T5 model:

.. code-block:: python

    def setup_model(model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer =  T5Tokenizer.from_pretrained(model_name)
        return model, tokenizer

We also, add couple of helper functions here for date and formatting memory
metrics.

.. code-block:: python

    def get_date_of_run():
        """create date and time for file save uniqueness
        example: 2022-05-07-08:31:12_PM'
        """
        date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
        print(f"--> current date and time of run = {date_of_run}")
        return date_of_run

    def format_metrics_to_gb(item):
        """quick function to format numbers to gigabyte and round to 4 digit precision"""
        metric_num = item / g_gigabyte
        metric_num = round(metric_num, ndigits=4)
        return metric_num


2.2 Define a train function:

.. code-block:: python

    def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
        model.train()
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(2).to(local_rank)

        if sampler:
            sampler.set_epoch(epoch)
        if rank==0:
            inner_pbar = tqdm.tqdm(
                range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
            )
        for batch in train_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            optimizer.zero_grad()
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch)
            if rank==0:
                inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_accuracy = fsdp_loss[0] / fsdp_loss[1]


        if rank == 0:
            inner_pbar.close()
            print(
                    f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
                )
        return train_accuracy

2.3 Define a validation function:

.. code-block:: python

    def validation(model, rank, world_size, val_loader):
        model.eval()
        correct = 0
        local_rank = int(os.environ['LOCAL_RANK'])
        fsdp_loss = torch.zeros(3).to(local_rank)
        if rank == 0:
            inner_pbar = tqdm.tqdm(
                range(len(val_loader)), colour="green", desc="Validation Epoch"
            )
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
                fsdp_loss[0] += output["loss"].item()  # sum up batch loss
                fsdp_loss[1] += len(batch)

                if rank==0:
                    inner_pbar.update(1)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        val_loss = fsdp_loss[0] / fsdp_loss[1]
        if rank == 0:
            inner_pbar.close()
            print(f"Validation Loss: {val_loss:.4f}")
        return val_loss


2.4 Define a distributed train function that wraps the model in FSDP:


.. code-block:: python


    def fsdp_main(args):

        model, tokenizer = setup_model("t5-base")

        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])


        dataset = load_dataset('wikihow', 'all', data_dir='data/')
        print(dataset.keys())
        print("Size of train dataset: ", dataset['train'].shape)
        print("Size of Validation dataset: ", dataset['validation'].shape)


        #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
        train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)
        val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

        sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

        setup()


        train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
        test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
        cuda_kwargs = {'num_workers': 2,
                        'pin_memory': True,
                        'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

        t5_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                T5Block,
            },
        )
        sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
        torch.cuda.set_device(local_rank)


        #init_start_event = torch.cuda.Event(enable_timing=True)
        #init_end_event = torch.cuda.Event(enable_timing=True)

        #init_start_event.record()

        bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
        )

        if bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None # defaults to fp32

        # model is on CPU before input to FSDP
        model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=mp_policy,
            #sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device())

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        best_val_loss = float("inf")
        curr_val_loss = float("inf")
        file_save_name = "T5-model-"

        if rank == 0:
            time_of_run = get_date_of_run()
            dur = []
            train_acc_tracking = []
            val_acc_tracking = []
            training_start_time = time.time()

        if rank == 0 and args.track_memory:
            mem_alloc_tracker = []
            mem_reserved_tracker = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
            if args.run_validation:
                curr_val_loss = validation(model, rank, world_size, val_loader)
            scheduler.step()

            if rank == 0:

                print(f"--> epoch {epoch} completed...entering save and stats zone")

                dur.append(time.time() - t0)
                train_acc_tracking.append(train_accuracy.item())

                if args.run_validation:
                    val_acc_tracking.append(curr_val_loss.item())

                if args.track_memory:
                    mem_alloc_tracker.append(
                        format_metrics_to_gb(torch.cuda.memory_allocated())
                    )
                    mem_reserved_tracker.append(
                        format_metrics_to_gb(torch.cuda.memory_reserved())
                    )
                print(f"completed save and stats zone...")

            if args.save_model and curr_val_loss < best_val_loss:

                # save
                if rank == 0:
                    print(f"--> entering save model state")

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state = model.state_dict()
                #print(f"saving process: rank {rank}  done w state_dict")


                if rank == 0:
                    print(f"--> saving model ...")
                    currEpoch = (
                        "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                    )
                    print(f"--> attempting to save model prefix {currEpoch}")
                    save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                    print(f"--> saving as model name {save_name}")

                    torch.save(cpu_state, save_name)

            if curr_val_loss < best_val_loss:

                best_val_loss = curr_val_loss
                if rank==0:
                    print(f"-->>>> New Val Loss Record: {best_val_loss}")

        dist.barrier()
        cleanup()


2.5 Parse the arguments and set the main function:

.. code-block:: python


    if __name__ == '__main__':
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
        parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=2, metavar='N',
                            help='number of epochs to train (default: 3)')
        parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                            help='learning rate (default: .002)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--track_memory', action='store_false', default=True,
                            help='track the gpu memory')
        parser.add_argument('--run_validation', action='store_false', default=True,
                            help='running the validation')
        parser.add_argument('--save-model', action='store_false', default=True,
                            help='For Saving the current Model')
        args = parser.parse_args()

        torch.manual_seed(args.seed)

        fsdp_main(args)


To run the the training using torchrun:

.. code-block:: bash

    torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

.. _transformer_wrapping_policy:

Transformer Wrapping Policy
---------------------------

As discussed in the `previous tutorial
<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__,
auto_wrap_policy is one of the FSDP features that make it easy to automatically
shard a given model and put the model, optimizer and gradient shards into
distinct FSDP units.

For some architectures such as Transformer encoder-decoders, some parts of the
model such as embedding table is being shared with both encoder and decoder.  In
this case, we need to place the embedding table in the outer FSDP unit so that
it could be accessed from both encoder and decoder.  In addition, by registering
the layer class for a transformer, the sharding plan can be made much more
communication efficient.  In PyTorch 1.12, FSDP added this support and now we
have a wrapping policy for transfomers.

It can be created as follows, where the T5Block represents the T5 transformer
layer class (holding MHSA and FFN).


.. code-block:: python

    t5_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                T5Block,
            },
        )
    torch.cuda.set_device(local_rank)


    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy)

To see the wrapped model, you can easily print the model and visually inspect
the sharding and FSDP units as well.


Mixed Precision
---------------
FSDP supports flexible mixed precision training allowing for arbitrary reduced
precision types (such as fp16 or bfloat16). Currently BFloat16 is only available
on Ampere GPUs, so you need to confirm native support before you use it. On
V100s for example, BFloat16 can still be run but because it runs non-natively,
it can result in significant slowdowns.

To check if BFloat16 is natively supported, you can use the following :

.. code-block:: python

    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

One of the advantages of mixed precision in FSDP is providing granular control
over different precision levels for parameters, gradients, and buffers as
follows:

.. code-block:: python

    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    fp32_policy = MixedPrecision(
        param_dtype=torch.float32,
        # Gradient communication precision.
        reduce_dtype=torch.float32,
        # Buffer precision.
        buffer_dtype=torch.float32,
    )

Note that if a certain type (parameter, reduce, buffer) is not specified, they
will not be casted at all.

This flexibility allows users fine grained control, such as only setting
gradient communication to happen in reduced precision, and all parameters /
buffer computation to be done in full precision. This is potentially useful in
cases where intra-node communication is the main bottleneck and parameters /
buffers must be in full precision to avoid accuracy issues. This can be done
with the following policy:

.. code-block:: bash

    grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)


In 2.4 we just add the relevant mixed precision policy to the FSDP wrapper:


.. code-block:: python

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen)

In our experiments, we have observed up to 4x speed up by using BFloat16 for
training and memory reduction of approximately 30% in some experiments that can
be used for batch size increases.


Intializing FSDP Model on Device
--------------------------------
In 1.12, FSDP supports a `device_id` argument meant to initialize input CPU
module on the device given by `device_id`. This is useful when the entire model
does not fit on a single GPU, but fits in a host's CPU memory. When `device_id`
is specified, FSDP will move the model to the specified device on a per-FSDP
unit basis, avoiding GPU OOM issues while initializing several times faster than
CPU-based initialization:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device())



Sharding Strategy
-----------------
FSDP sharding strategy by default is set to fully shard the model parameters,
gradients and optimizer states get sharded across all ranks. (also termed Zero3
sharding). In case you are interested to have the Zero2 sharding strategy, where
only optimizer states and gradients are sharded, FSDP support this feature by
passing the Sharding strategy by using  "ShardingStrategy.SHARD_GRAD_OP",
instead of "ShardingStrategy.FULL_SHARD" to the FSDP initialization  as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # ZERO2)

This will reduce the communication overhead in FSDP, in this case, it holds full
parameters after forward and through the backwards pass.

This saves an all_gather during backwards so there is less communication at the
cost of a higher memory footprint. Note that full model params are freed at the
end of backwards and all_gather will happen on the next forward pass.

Backward Prefetch
-----------------
The backward prefetch setting controls the timing of when the next FSDP unit's
parameters should be requested.  By setting it to `BACKWARD_PRE`, the next
FSDP's unit params can begin to be requested and arrive sooner before the
computation of the current unit starts. This overlaps the `all_gather`
communication and gradient computation which can increase the training speed in
exchange for slightly higher memory consumption. It can be utilized in the FSDP
wrapper in 2.4 as follows:

.. code-block:: python

    torch.cuda.set_device(local_rank)

     model = FSDP(model,
            auto_wrap_policy=t5_auto_wrap_policy,
            mixed_precision=bfSixteen,
            device_id=torch.cuda.current_device(),
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE)

`backward_prefetch` has two modes, `BACKWARD_PRE` and `BACKWARD_POST`.
`BACKWARD_POST` means that the next FSDP unit's params will not be requested
until the current FSDP unit processing is complete, thus minimizing memory
overhead.  In some cases, using `BACKWARD_PRE` can increase model training speed
up to 2-10%, with even higher speed improvements noted for larger models.

Model Checkpoint Saving, by streaming to the Rank0 CPU
------------------------------------------------------
To save model checkpoints using FULL_STATE_DICT saving which saves model in the
same fashion as a local model, PyTorch 1.12 offers a few utilities to support
the saving of larger models.

First, a FullStateDictConfig can be specified, allowing the state_dict to be
populated on rank 0 only and offloaded to the CPU.

When using this configuration, FSDP will allgather model parameters, offloading
them to the CPU one by one, only on rank 0. When the state_dict is finally
saved, it will only be populated on rank 0 and contain CPU tensors. This avoids
potential OOM for models that are larger than a single GPU memory and allows
users to checkpoint models whose size is roughly the available CPU RAM on the
user's machine.

This feature can be run as follows:

.. code-block:: python

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
    if rank == 0:
     save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
     torch.save(cpu_state, save_name)

Summary
-------

In this tutorial, we have introduced many new features for FSDP available in
Pytorch 1.12 and used HF T5 as the running example.  Using the proper wrapping
policy especially for transformer models, along with mixed precision and
backward prefetch should speed up your training runs. Also, features such as
initializing the model on device, and checkpoint saving via streaming to CPU
should help to avoid OOM error in dealing with large models.

We are actively working to add new features to FSDP for the next release. If
you have feedback, feature requests, questions or are encountering issues
using FSDP, please feel free to contact us by opening an issue in the
`PyTorch Github repository <https://github.com/pytorch/pytorch>`__.