Great question! This gets to the heart of why we need multiple parallelism strategies.

## Why TP is Needed When We Have FSDP

FSDP has a fundamental limitation: **it can only shard parameters, but individual operations still need to fit in GPU memory**.

Here's the key issue:

```python
# In FSDP, when computing a layer:
# 1. All-gather the sharded weights for this layer
# 2. Compute: output = input @ weight  # This computation must fit in memory!
# 3. Discard the weights after computation
```

**The problem**: Even though FSDP shards the weights across devices, during the actual forward pass, each GPU needs to temporarily hold the entire weight matrix for that layer in memory to do the computation.

**Example scenario**:
- You have a 70B parameter model
- One attention layer has a 8192 x 32768 weight matrix (1GB)
- Your GPU only has 24GB memory
- FSDP can shard this across 8 GPUs (125MB per GPU when sharded)
- But during forward pass, each GPU needs the full 1GB matrix temporarily
- If your activation memory + other overhead is already using 23GB, you're out of memory!

## How TP is Different from FSDP

| Aspect | FSDP | Tensor Parallelism |
|--------|------|-------------------|
| **What's sharded** | Parameters are sharded, gathered for computation | Computation itself is sharded |
| **Memory during compute** | Full layer weights needed temporarily | Only local shard needed |
| **Communication pattern** | All-gather → compute → discard | Compute locally → all-reduce results |
| **Activation handling** | Full activations on each device | Activations split across devices |

## Visual Example

**FSDP approach**:
```
GPU 0: Has W_shard_0, needs to gather full W for computation
GPU 1: Has W_shard_1, needs to gather full W for computation  
GPU 2: Has W_shard_2, needs to gather full W for computation
GPU 3: Has W_shard_3, needs to gather full W for computation

Forward pass: Each GPU computes Y = X @ W_full (memory spike!)
```

**TP approach**:
```
GPU 0: Always only holds W_0, computes Y_0 = X @ W_0
GPU 1: Always only holds W_1, computes Y_1 = X @ W_1
GPU 2: Always only holds W_2, computes Y_2 = X @ W_2  
GPU 3: Always only holds W_3, computes Y_3 = X @ W_3

Forward pass: All-reduce [Y_0, Y_1, Y_2, Y_3] → Y_final
```

## When You Need Both

For the largest models, you often use **hybrid parallelism**:
- **TP within nodes** (2-8 GPUs with fast interconnect)
- **FSDP across nodes** (for data parallelism)
- **Pipeline parallelism** for even larger models

This is why libraries like DeepSpeed and FairScale support combining multiple parallelism strategies - no single approach solves all the memory and computation challenges of modern large models.

The key insight: FSDP solves parameter storage, but TP solves computation memory constraints.

The memory spike happens in **both forward AND backward passes** with FSDP!

## Forward Pass Memory Spike
```python
# FSDP forward pass
def forward(self, x):
    # 1. All-gather sharded weights (MEMORY SPIKE!)
    full_weight = all_gather(self.weight_shard)  # Temporary full weight in memory
    
    # 2. Compute
    output = x @ full_weight
    
    # 3. Free the gathered weight
    del full_weight  # Memory freed
    
    return output
```

## Backward Pass Memory Spike  
```python
# FSDP backward pass
def backward(self, grad_output):
    # 1. All-gather weights AGAIN (ANOTHER MEMORY SPIKE!)
    full_weight = all_gather(self.weight_shard)  # Need weights for gradient computation
    
    # 2. Compute gradients
    grad_input = grad_output @ full_weight.T
    grad_weight = input.T @ grad_output
    
    # 3. Reduce-scatter the weight gradients
    grad_weight_shard = reduce_scatter(grad_weight)
    
    # 4. Free the gathered weight
    del full_weight  # Memory freed again
    
    return grad_input
```

## Why Both Passes Need Full Weights

- **Forward pass**: Need full weight matrix to compute `output = input @ weight`
- **Backward pass**: Need full weight matrix to compute:
  - Input gradients: `grad_input = grad_output @ weight.T` 
  - Weight gradients: `grad_weight = input.T @ grad_output`

## Memory Timeline

```
Time →
Forward:  [normal] → [SPIKE: gather weights] → [compute] → [free] → [normal]
Backward: [normal] → [SPIKE: gather weights] → [compute] → [free] → [normal]
```

## This is Why TP is Still Needed

Even with FSDP's parameter sharding, you get these periodic memory spikes that can cause OOM. With tensor parallelism:

```python
# TP - no memory spikes, weights always stay local
def forward(self, x):
    local_output = x @ self.local_weight_shard  # No gathering needed
    final_output = all_reduce(local_output)     # Much smaller communication
    return final_output
```

This is exactly why the largest models often use hybrid approaches - FSDP reduces baseline memory usage, but TP eliminates the memory spikes during computation.

Great question! Let me break down column-wise vs row-wise sharding with concrete examples.

## Matrix Dimensions First

Let's say we have a FeedForward layer with:
- Input dimension: 4096 (hidden size)
- Intermediate dimension: 11008 (usually 4x hidden size)
- 4 GPUs for tensor parallelism

So our weight matrices are:
- `w1`: [4096, 11008] - projects input to intermediate
- `w2`: [11008, 4096] - projects intermediate back to hidden
- `w3`: [4096, 11008] - second projection (for SwiGLU)

## Column-wise Parallel (w1 and w3)

**Column-wise = split along the column dimension**

```python
# Original w1: [4096, 11008]
# Split into 4 parts along columns:

GPU 0: w1_part0 = [4096, 2752]  # columns 0:2752
GPU 1: w1_part1 = [4096, 2752]  # columns 2752:5504  
GPU 2: w1_part2 = [4096, 2752]  # columns 5504:8256
GPU 3: w1_part3 = [4096, 2752]  # columns 8256:11008

# Same for w3: [4096, 11008] split into 4 parts
GPU 0: w3_part0 = [4096, 2752]
GPU 1: w3_part1 = [4096, 2752] 
GPU 2: w3_part2 = [4096, 2752]
GPU 3: w3_part3 = [4096, 2752]
```

## Forward Pass with Column-wise Sharding

```python
# Input x: [batch_size, seq_len, 4096] - replicated on all GPUs
# Each GPU computes its portion:

GPU 0: 
  out1_0 = x @ w1_part0  # [batch, seq, 2752]
  out3_0 = x @ w3_part0  # [batch, seq, 2752]
  intermediate_0 = F.silu(out1_0) * out3_0  # [batch, seq, 2752]

GPU 1:
  out1_1 = x @ w1_part1  # [batch, seq, 2752] 
  out3_1 = x @ w3_part1  # [batch, seq, 2752]
  intermediate_1 = F.silu(out1_1) * out3_1  # [batch, seq, 2752]

# ... same for GPU 2 and 3
```

After this step, each GPU has a **partial result** that represents different "neurons" or "features" in the intermediate layer.

## Row-wise Parallel (w2)

**Row-wise = split along the row dimension**

```python
# Original w2: [11008, 4096]
# Split into 4 parts along rows:

GPU 0: w2_part0 = [2752, 4096]  # rows 0:2752
GPU 1: w2_part1 = [2752, 4096]  # rows 2752:5504
GPU 2: w2_part2 = [2752, 4096]  # rows 5504:8256  
GPU 3: w2_part3 = [2752, 4096]  # rows 8256:11008
```

## Row-wise Computation

```python
# Each GPU multiplies its intermediate result with its w2 part:

GPU 0: partial_out_0 = intermediate_0 @ w2_part0  # [batch, seq, 4096]
GPU 1: partial_out_1 = intermediate_1 @ w2_part1  # [batch, seq, 4096]  
GPU 2: partial_out_2 = intermediate_2 @ w2_part2  # [batch, seq, 4096]
GPU 3: partial_out_3 = intermediate_3 @ w2_part3  # [batch, seq, 4096]
```

## The Magic: Why Only One All-Reduce

Here's the key insight - each GPU's `partial_out` represents the **same output dimensions** but with **partial contributions**:

```python
# The final result is the SUM of all partial outputs:
final_output = partial_out_0 + partial_out_1 + partial_out_2 + partial_out_3

# This is exactly what all-reduce does!
final_output = all_reduce([partial_out_0, partial_out_1, partial_out_2, partial_out_3])
```

## Visual Example

```
Input x: [batch, seq, 4096] (replicated on all GPUs)

Column-wise w1, w3:
┌─────────────────────────────────────────────┐
│  w1: [4096, 11008]                         │
│  ┌──────┬──────┬──────┬──────┐              │
│  │ part0│ part1│ part2│ part3│              │
│  │ 2752 │ 2752 │ 2752 │ 2752 │              │
│  │  cols│  cols│  cols│  cols│              │
│  └──────┴──────┴──────┴──────┘              │
└─────────────────────────────────────────────┘

After w1, w3 computation:
GPU 0: [batch, seq, 2752] intermediate_0
GPU 1: [batch, seq, 2752] intermediate_1  
GPU 2: [batch, seq, 2752] intermediate_2
GPU 3: [batch, seq, 2752] intermediate_3

Row-wise w2:
┌─────────────────────────┐
│  w2: [11008, 4096]      │
│  ┌─────────────────────┐ │
│  │ part0: [2752, 4096] │ │ GPU 0
│  ├─────────────────────┤ │
│  │ part1: [2752, 4096] │ │ GPU 1
│  ├─────────────────────┤ │  
│  │ part2: [2752, 4096] │ │ GPU 2
│  ├─────────────────────┤ │
│  │ part3: [2752, 4096] │ │ GPU 3
│  └─────────────────────┘ │
└─────────────────────────┘

Final step - All-reduce:
GPU 0: partial_out_0 ┐
GPU 1: partial_out_1 ├─ All-reduce ─→ final_output [batch, seq, 4096]
GPU 2: partial_out_2 │
GPU 3: partial_out_3 ┘
```

## Why This Pattern Works

1. **Column-wise sharding** naturally splits the "expansion" phase - each GPU handles different intermediate features
2. **Row-wise sharding** naturally splits the "contraction" phase - each GPU contributes partial sums to the final output
3. **Mathematical equivalence**: `(x @ w1) @ w2 = x @ (w1 @ w2)` - we're just computing it in a distributed way
4. **Single communication**: Only need one all-reduce at the very end instead of gathering tensors multiple times

This is why this specific sharding pattern is so elegant - it follows the natural flow of the computation while minimizing communication!

You're absolutely right to notice this! Both TP and FSDP do divide weight matrices, but they divide them in **fundamentally different ways** and for **different purposes**. Let me clarify the key differences:

## Weight Matrix Division: TP vs FSDP

### Tensor Parallelism (TP)
```python
# TP divides for COMPUTATION
# w1: [4096, 11008] → split along columns

GPU 0: w1_part0 = [4096, 2752]  # columns 0:2752
GPU 1: w1_part1 = [4096, 2752]  # columns 2752:5504
GPU 2: w1_part2 = [4096, 2752]  # columns 5504:8256
GPU 3: w1_part3 = [4096, 2752]  # columns 8256:11008

# During forward pass:
GPU 0: out_0 = x @ w1_part0  # Each GPU computes DIFFERENT parts
GPU 1: out_1 = x @ w1_part1  # of the SAME computation
GPU 2: out_2 = x @ w1_part2
GPU 3: out_3 = x @ w1_part3
```

### FSDP
```python
# FSDP divides for STORAGE
# w1: [4096, 11008] → split arbitrarily (e.g., by flattened index)

GPU 0: w1_shard0 = [1024, 2752]  # arbitrary chunk of parameters
GPU 1: w1_shard1 = [1024, 2752]  # another arbitrary chunk
GPU 2: w1_shard2 = [1024, 2752]  # another arbitrary chunk  
GPU 3: w1_shard3 = [1024, 2752]  # final chunk

# During forward pass:
# ALL GPUs gather to get full w1: [4096, 11008]
full_w1 = all_gather([w1_shard0, w1_shard1, w1_shard2, w1_shard3])
GPU 0: out = x @ full_w1  # Each GPU computes the SAME computation
GPU 1: out = x @ full_w1  # with the SAME full matrix
GPU 2: out = x @ full_w1
GPU 3: out = x @ full_w1
```

## Key Differences

| Aspect | Tensor Parallelism | FSDP |
|--------|-------------------|------|
| **Purpose** | Divide COMPUTATION | Divide STORAGE |
| **Division Strategy** | Semantically meaningful (columns/rows) | Arbitrary chunks for storage |
| **During Computation** | Each GPU computes DIFFERENT parts | Each GPU computes SAME thing |
| **Memory During Forward** | Only local shard needed | Full matrix temporarily needed |
| **Communication Pattern** | All-reduce final results | All-gather weights, then compute |

## Visual Comparison

**TP Computation Flow:**
```
Input x (replicated) → Each GPU computes different output slices → All-reduce → Final output
```

**FSDP Computation Flow:**
```
Sharded weights → All-gather full weights → All GPUs compute same output → Discard weights
```

## The Critical Difference

**TP**: "Let's divide the work - you compute columns 0-2752, I'll compute columns 2752-5504"
**FSDP**: "Let's divide the storage - you store part 1, I'll store part 2, but when we need to compute, we'll gather everything and all do the same math"

This is why:
- **TP** eliminates memory spikes (no gathering needed)
- **FSDP** has memory spikes (must gather full tensors)
- **TP** requires specific mathematical patterns (column/row-wise splits)
- **FSDP** can split any parameter arbitrarily

You're correct that both divide weight matrices, but TP divides them **mathematically** while FSDP divides them **logistically**!


Excellent question! You're thinking about the dependencies correctly. Let me clarify what happens with w2 and how the row-wise sharding works.

## The Key: w2 is Also Sharded (Row-wise)

Remember, w2 is sharded **row-wise** across the same GPUs:

```python
# Original w2: [11008, 4096]
# Split into 4 parts along ROWS:

GPU 0: w2_part0 = [2752, 4096]  # rows 0:2752
GPU 1: w2_part1 = [2752, 4096]  # rows 2752:5504  
GPU 2: w2_part2 = [2752, 4096]  # rows 5504:8256
GPU 3: w2_part3 = [2752, 4096]  # rows 8256:11008
```

## The Beautiful Alignment

Each GPU has **matching dimensions**:

```python
GPU 0: 
  intermediate_0: [batch, seq, 2752]  # Output from w1/w3
  w2_part0:      [2752, 4096]        # Matching input dimension!
  
GPU 1:
  intermediate_1: [batch, seq, 2752]  # Output from w1/w3  
  w2_part1:      [2752, 4096]        # Matching input dimension!

# And so on...
```

## The Row-wise Computation

```python
# Each GPU computes with its own w2 part:
GPU 0: partial_out_0 = intermediate_0 @ w2_part0  # [batch, seq, 4096]
GPU 1: partial_out_1 = intermediate_1 @ w2_part1  # [batch, seq, 4096]
GPU 2: partial_out_2 = intermediate_2 @ w2_part2  # [batch, seq, 4096]  
GPU 3: partial_out_3 = intermediate_3 @ w2_part3  # [batch, seq, 4096]
```

## Why This Works Mathematically

The key insight is that this is equivalent to:

```python
# What we're computing in distributed fashion:
# intermediate = [intermediate_0, intermediate_1, intermediate_2, intermediate_3]  # concat
# full_w2 = [w2_part0; w2_part1; w2_part2; w2_part3]  # stack vertically

# Matrix multiplication distributes over concatenation:
# intermediate @ full_w2 = 
#   intermediate_0 @ w2_part0 + 
#   intermediate_1 @ w2_part1 + 
#   intermediate_2 @ w2_part2 + 
#   intermediate_3 @ w2_part3

# That's exactly what we compute separately and then all-reduce!
```

## Visual Representation

```
Column-wise w1/w3:           Row-wise w2:
┌──────┬──────┬──────┬──────┐   ┌─────────────────────┐
│ w1_0 │ w1_1 │ w1_2 │ w1_3 │   │ w2_0: [2752, 4096] │ ← GPU 0
└──────┴──────┴──────┴──────┘   ├─────────────────────┤
   ↓      ↓      ↓      ↓       │ w2_1: [2752, 4096] │ ← GPU 1  
GPU 0   GPU 1  GPU 2  GPU 3     ├─────────────────────┤
   ↓      ↓      ↓      ↓       │ w2_2: [2752, 4096] │ ← GPU 2
[2752]  [2752] [2752] [2752]    ├─────────────────────┤
                                │ w2_3: [2752, 4096] │ ← GPU 3
                                └─────────────────────┘

Perfect alignment! Each GPU's intermediate output
matches its w2 part's input dimension.
```

## No Communication Needed Until Final Step

The beautiful thing is:
1. **No communication** needed between w1/w3 and w2 computations
2. Each GPU has everything it needs locally
3. Only **one all-reduce** at the very end to sum the partial results

This is why this sharding pattern is so efficient - the column-wise and row-wise splits are **perfectly coordinated** so that data flows naturally without intermediate communication!


Here’s a simplified walkthrough of why, in Megatron-style tensor parallelism, you’d shard the *first* two feed-forward matrices (“w1” and “w3”) **column-wise** but shard the *final* matrix (“w2”) **row-wise**.

---

## 1. The Feed-Forward Layer at a Glance

A typical “SwiGLU”-style MLP block in a Transformer does something like:

```python
# x: [batch, D_model]
h1 = w1(x)         # → [batch, D_hidden]
h2 = w3(x)         # → [batch, D_hidden]
m = h1 * F.silu(h2) # element-wise, still [batch, D_hidden]
out = w2(m)        # → [batch, D_model]
```

Here each `wi` is a Linear layer:

* **w1**: `[D_model → D_hidden]`
* **w3**: `[D_model → D_hidden]`
* **w2**: `[D_hidden → D_model]`

---

## 2. Column-Parallel for w1 & w3

**Column parallelism** means splitting the weight’s **output** dimension (its columns) across GPUs:

* Suppose `D_hidden = 8` and we have 2 GPUs. We split into two halves of 4:

  * GPU0 holds `w1[:, :4]`  & GPU1 holds `w1[:, 4:]`
  * Likewise for `w3`.

### Why column-wise here?

1. **No reduction needed**: Each GPU computes its chunk of the hidden features; the final hidden vector is just the **concatenation** of those chunks across GPUs.
2. **Parallel w1 & w3 together**: Since both project from `D_model → D_hidden`, you can apply the same split for both, doing both mat-muls locally with no inter-GPU communication until concatenation.
3. **Efficient for wide layers**: MLP hidden dims are often large, so splitting columns balances compute and storage.

```
x: [B, D_model]
 on GPU0  on GPU1
w1 [:, :4]  w1[:, 4:]
 → y1=[B,4]   → y2=[B,4]
concat → [B,8]   ← h1
(same for w3)
```

---

## 3. Row-Parallel for w2

**Row parallelism** means splitting the weight’s **input** dimension (its rows) across GPUs:

* Here `w2` is `[D_hidden → D_model]`. With `D_hidden=8`, 2 GPUs each hold 4 rows:

  * GPU0 has `w2[:4, :]`
  * GPU1 has `w2[4:, :]`

### Why row-wise here?

1. **Partial sums**: Each GPU multiplies its **portion** of the hidden vector `m` (split correspondingly) against its rows of `w2`, yielding a **partial output** of shape `[B, D_model]`.
2. **All-reduce needed**: To get the true final `[B, D_model]` activation, you sum (all-reduce) those partial outputs across GPUs.
3. **No need to concatenate**: Unlike column parallel, the outputs here share the same output dimension and must be **summed**, not concatenated.

```
m: [B,8]  split into m1=[B,4] and m2=[B,4]
 on GPU0             on GPU1
w2[:4, :]              w2[4:, :]
 m1 × w2[:4] → y1=[B,D_model]
 m2 × w2[4:] → y2=[B,D_model]
 all-reduce sum(y1, y2) → [B,D_model]
```

---

## 4. Summary of Communication Patterns

| Layer  | Weight Shape         | Split Dim            | Local Compute                     | Aggregation        |
| ------ | -------------------- | -------------------- | --------------------------------- | ------------------ |
| w1, w3 | `[D_model×D_hidden]` | **columns** (hidden) | compute `[B, D_hidden/​G]` chunks | **concatenate**    |
| w2     | `[D_hidden×D_model]` | **rows** (hidden)    | compute `[B, D_model]` partials   | **all-reduce sum** |

* **Column-parallel** lets you shard wide output features with **no cross-GPU reduction**, only a final concatenation.
* **Row-parallel** shards input features, producing partial outputs that must be **summed** across GPUs.

This design minimizes the number of cross-GPU collectives while balancing memory and compute loads.

---

**References**

* Megatron-LM paper: Tensor parallelism for Transformer layers ([arxiv.org][1])
* Megatron-LM config examples: column/row splits explained ([apxml.com][2])

[1]: https://arxiv.org/pdf/1909.08053?utm_source=chatgpt.com "Megatron-LM: Training Multi-Billion Parameter Language Models Using ..."
[2]: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks/configuring-megatron-lm?utm_source=chatgpt.com "Configuring Megatron-LM Parallelism - apxml.com"


In tensor-parallel (TP) schemes like Megatron-LM, the choice between column-wise and row-wise sharding boils down to **which tensor dimension you want each device to own locally**, and **how the downstream computation aggregates those partial results**. Here’s a rule-of-thumb:

1. **Column-wise (output) sharding**

   * **What it splits**: the **output feature** dimension (the “columns” of the weight matrix).
   * **When to use**: if after your local mat-mul each device can simply **concatenate** its partial outputs along that same dimension to form the full activation—no cross-device reduction required.
   * **Typical layers**: the *first* projection(s) in an MLP or self-attention Q/K/V that fan out into a large hidden space.

2. **Row-wise (input) sharding**

   * **What it splits**: the **input feature** dimension (the “rows” of the weight matrix).
   * **When to use**: if each device multiplies its slice of the input features against its full set of output channels but the final outputs need an **all-reduce sum** across devices to accumulate the contributions from every input shard.
   * **Typical layers**: the *final* projection in an MLP (back to model dimension) or the output projection in self-attention (from hidden back to model dim).

---

## Why this pattern makes sense

* **Minimize communication**

  * Column-wise → only a final **concat** (cheap, local)
  * Row-wise → one **all-reduce** per batch (more expensive, but only once)

* **Balance compute & memory**

  * Splitting the **largest** dimension of a wide layer maximizes per-GPU compute while fitting parameters/activations in memory.

* **Compatibility with activation flow**

  * Early layers produce large hidden representations—concatenation is natural.
  * Late layers collapse back to model dimension—partial sums must be reduced.

---

## Applying to your mapping

```python
layer_tp_plan = {
    # Self-attention
    "attention.wq": ColwiseParallel(),   # fan-out into many heads → concat queries
    "attention.wk": ColwiseParallel(),   # same for keys
    "attention.wv": ColwiseParallel(),   # same for values
    "attention.wo": RowwiseParallel(),   # combine heads back → sum

    # Feed-forward (SwiGLU style)
    "feed_forward.w1": ColwiseParallel(), # project ➔ large hidden → concat
    "feed_forward.w3": ColwiseParallel(), # same
    "feed_forward.w2": RowwiseParallel(), # collapse hidden ➔ model dim → sum
}
```

* **w1 & w3** both map from `[model_dim → hidden_dim]`. Splitting **hidden\_dim** (columns) means each GPU computes a subset of hidden features, and you can simply concatenate them.
* **w2** maps from `[hidden_dim → model_dim]`. Splitting **hidden\_dim** (rows) means each GPU handles part of the hidden vector, and then you need to all-reduce their partial contributions to the model-dim output.

---

### General guideline checklist

1. **What shape is your weight?** `W ∈ ℝ[input_dim × output_dim]`

2. **Can you concatenate partial outputs?**

   * Yes → split **output\_dim** (column-wise).
   * No → split **input\_dim** (row-wise) and then **reduce\_sum**.

3. **Look at the downstream operation**: if it’s a simple reshape/concat, go column; if it’s an elementwise add or sum across shards, go row.

That simple rule will cover virtually all linear/projection layers in Transformers.

| Layer Type                             | Weight Shape        | Split Strategy | Why?                                                                                    |
| -------------------------------------- | ------------------- | -------------- | --------------------------------------------------------------------------------------- |
| **Self-Attention Q/K/V**               | (d\_model↦d\_head)  | Column-wise    | Each head’s queries/keys/values independent; concat later. ([minjiazhang.github.io][1]) |
| **Attention Output (Wo)**              | (d\_model↦d\_model) | Row-wise       | Must sum contributions from all heads. ([developer.download.nvidia.com][2])             |
| **FFN First Linear (W1)**              | (d\_model↦d\_ff)    | Column-wise    | Produces separate hidden activations per shard. ([minjiazhang.github.io][1])            |
| **FFN Second Linear (W3)**             | (d\_model↦d\_ff)    | Column-wise    | Same rationale as W1. ([minjiazhang.github.io][1])                                      |
| **FFN Output Linear (W2)**             | (d\_ff↦d\_model)    | Row-wise       | Needs sum of all hidden contributions. ([developer.download.nvidia.com][2])             |
| **Final Classifier / LM Head**         | (d\_model↦vocab)    | Row-wise       | Summation across model dims for each vocab logit.                                       |
| **Embedding Layer (token embeddings)** | (vocab↦d\_model)    | Column-wise    | Different tokens independent; gather embeddings later.                                  |

[1]: https://minjiazhang.github.io/courses/sp24-resource/Megatron-LM.pdf?utm_source=chatgpt.com "Megatron-LM Training Multi-billion Parameter Language Models Using ..."
[2]: https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21496-megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.pdf?utm_source=chatgpt.com "MEGATRON-LM: TRAINING BILLION PARAMETER LANGUAGE MODELS WITH ... - Nvidia"


The general pattern
Column-wise shard a projection ⇒ output tensor’s last dim is now (original_dim // world_size).

Any reshape / view that assumes the original dim must be rewritten to use the local size.

After any elementwise or independent-per-head ops, you either concatenate or all-gather back to full shape before the next fused step (e.g., before wo which is row-sharded).

When you apply **column-wise sharding** to a Linear layer, you’re slicing its **output** dimension across devices. In a Transformer’s attention block (like LLaMA), after the query/key/value projections (`wq`, `wk`, `wv`) we typically do something like:

```python
# unsharded:
x_proj = Wq @ x         # shape [batch, seq, heads * head_dim]
x_proj = x_proj.view(batch, seq, heads, head_dim)
```

—but if `Wq` is sharded **column-wise** across, say, 4 GPUs, then each GPU only holds and computes a “quarter” of those output columns. Concretely:

* **Unsharded** `Wq` is `[embed_dim, heads * head_dim]`.
* **Sharded** `Wq` on 4 GPUs → each has `[embed_dim, (heads * head_dim) / 4]`.
* The local projection on each GPU is `[batch, seq, (heads * head_dim) / 4]`.

That last dimension is now **local\_heads × head\_dim**, not full `heads × head_dim`.

---

### Why adjust the `view`

Right after `wq(x)` we do a `.view(batch, seq, heads, head_dim)`. If you leave that unchanged, you’ll be telling PyTorch to interpret **only** your local slice as if it were the **full** `heads`:

```python
# Still wrong:
local_proj = local_proj.view(batch, seq, heads, head_dim)
```

But `local_proj` only has `heads/4` in that dimension! You need to reshape to **your** local head count:

```python
local_heads = heads // world_size
local_proj = local_proj.view(batch, seq, local_heads, head_dim)
```

Then, later on, the full attention can be **all-gathered** across head shards (or each shard attends on its local heads and the results are concatenated), and downstream code remains correct.

---

### The general pattern

1. **Column-wise shard a projection** ⇒ output tensor’s last dim is now `(original_dim // world_size)`.
2. **Any reshape / view** that assumes the original dim must be rewritten to use the **local** size.
3. **After any elementwise or independent-per-head ops**, you either concatenate or all-gather back to full shape before the next fused step (e.g., before `wo` which is row-sharded).

---

### Example in context

```python
# Suppose embed_dim=1024, heads=16, head_dim=64, world_size=4
# Unsharded wq: [1024, 16*64=1024]
# Sharded wq on 4 GPUs: each has [1024, 1024/4=256] = 4 heads × 64

local_proj = wq_local(x)                      # -> [batch, seq, 256]
local_heads = 16 // 4                         # = 4
# Correct view for this shard:
local_proj = local_proj.view(batch, seq, local_heads, head_dim)
# -> [batch, seq, 4, 64]
```

Without that adjustment, you’d try to view `[batch, seq, 256]` as `[batch, seq, 16, 64]` and get a runtime error or silently corrupt your data.

---

### Why it matters for LLaMA’s attention

LLaMA’s code does several tensor ops—`view`, `transpose`, `reshape`—to go from a flat `[batch, seq, heads*head_dim]` to the multi-headed shape and back. **If you insert a column-shard** without updating those shape ops, the indexing will be off. By updating every `heads` → `local_heads` in those reshapes, you ensure:

* Your shard sees exactly its slice of heads.
* Downstream all-gather or concatenate steps reconstruct the full multi-head tensor correctly.

---

### Key takeaway

Whenever you shard **along** a dimension that later participates in a reshape:

* **Identify** all the `view`/`reshape` calls that reference that original dimension.
* **Replace** the global size with the **local** size (`global // world_size`).
* **Ensure** that any later communication (all-gather, concat) reassembles the full tensor before you feed into layers that expect the complete shape.


Below is a **line-by-line unpacking** of the six code chunks I sent.
I keep the code in-line (grey boxes) so you can glance at it and then read the plain-English next to it.

---

## 1.  Spin-up distributed + make a TP mesh

```python
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")          # 1-A
tp_mesh = init_device_mesh("cuda", (8,)) # 1-B
```

| Step    | What really happens                                                                                                                                                                                                                                                                          |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1-A** | Starts NCCL communicators for **all ranks** in the job. That’s the same call you already use in DDP/FSDP.                                                                                                                                                                                    |
| **1-B** | Builds a `DeviceMesh` object that contains exactly **8 ranks** (one per local GPU). Internally PyTorch slices the world‐size list into a smaller process-group and tags it “tp”. All Tensor-Parallel collectives (all-gather / reduce-scatter / all-reduce) will *only* touch those 8 ranks. |

Think of `DeviceMesh` as “the set of GPUs that will share each weight”.

---

## 2.  Draft a Tensor-Parallel plan

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, SequenceParallel,
    PrepareModuleInput, parallelize_module
)

plan = {
    "attention.wq": ColwiseParallel(),  # 2-A
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),  # 2-B

    "feed_forward.w1": ColwiseParallel(), # 2-C
    "feed_forward.w3": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(), # 2-D

    "attention_norm": SequenceParallel(), # 2-E
    "ffn_norm":       SequenceParallel(),
}
```

| Tag       | Meaning                                                                                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2-A**   | Split the *columns* of the Q/K/V weight matrices across 8 GPUs. Each GPU computes 1/8th of the projection for its batch tokens.                                                         |
| **2-B**   | Split the *rows* of the output‐projection (`wo`). The rows align with the 1/8 pieces that came out of 2-A so only one final `all_reduce` is needed.                                     |
| **2-C/D** | Same trick for the SwiGLU MLP: two column-sharded FCs (`w1`, `w3`) feed into one row-sharded FC (`w2`).                                                                                 |
| **2-E**   | Tell PyTorch to keep the activations of the two `RMSNorm` layers **sharded on the sequence dimension**. That halves activation memory and postpones any gather until absolutely needed. |

Why bother writing this dict? -- Because `parallelize_module` (next section) reads it and rewrites the module *for you*.

---

## 3.  Parallelise every Transformer block

```python
for blk in model.layers:
    blk.attention.n_heads    //= tp_mesh.size()   # 3-A
    blk.attention.n_kv_heads //= tp_mesh.size()

    parallelize_module(blk, tp_mesh, plan)       # 3-B
```

| Tag     | What & Why                                                                                                                                                                                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3-A** | A Llama block defines `n_heads=32` (for example). After 8-way TP, each GPU holds 4 heads, so we divide. Same for key/value heads.                                                                                                                                   |
| **3-B** | `parallelize_module` walks the sub-modules, swaps each listed weight tensor into a `DTensor` (distributed tensor) with the right sharding layout, and injects tiny hooks before/after the module that fire the needed collectives. **No manual NCCL code** for you. |

Result: every weight lives on exactly one GPU slice; forward() still looks identical from user side.

---

## 4.  Handle embeddings, projector, and Loss Parallel

```python
model = parallelize_module(
    model, tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(output_layouts=Shard(1)), # 4-A
        "norm":           SequenceParallel(),                      # 4-B
        "output": ColwiseParallel(
            input_layouts=Shard(1),                                # 4-C
            use_local_output=False,                                # 4-D
        ),
    },
)

from torch.distributed.tensor.parallel import loss_parallel
import torch.nn.functional as F

with loss_parallel():                                 # 4-E
    out  = model(input_ids)
    loss = F.cross_entropy(out.flatten(0,1),
                           tgt.flatten(0,1))
    loss.backward()
```

| Tag     | Detail                                                                                                                                                                                                      |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **4-A** | The token embedding’s rows equal the vocab size (huge), so we split rows – each GPU stores 1/8-th of the vocab. Output activations are sharded on **sequence** (`Shard(1)`), so the next block can keep SP. |
| **4-B** | Final `RMSNorm` also uses Sequence Parallel.                                                                                                                                                                |
| **4-C** | The last linear projector takes seq-sharded activations (`Shard(1)`) and is column-sharded.                                                                                                                 |
| **4-D** | `use_local_output=False` tells PyTorch “keep the logits as a DTensor, not a regular tensor”, because we’re about to feed them to `loss_parallel`.                                                           |
| **4-E** | `loss_parallel()` is a context manager that rewrites cross-entropy kernels so they work on sharded logits. It avoids the massive all-gather that would normally merge `[B,S,V]` across GPUs.                |

Bottom line: we cut vocab-dim memory by 8 and skip a communication barrier.

---

## 5.  Marry TP with inter-host FSDP (2-D parallelism)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

mesh2d  = init_device_mesh("cuda", (dp, 8), mesh_dim_names=("dp","tp")) # 5-A
tp_mesh = mesh2d["tp"]   # 5-B
dp_mesh = mesh2d["dp"]

model   = parallelize_module(model, tp_mesh, plan)  # 5-C
model   = FSDP(model, device_mesh=dp_mesh,          # 5-D
               use_orig_params=True,
               sharding_strategy="HYBRID_SHARD")
```

| Tag     | Explanation                                                                                                                                                              |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **5-A** | Builds a 2-D grid: first dimension = data-parallel (inter-host), second = tensor-parallel (intra-host, 8 ranks). Example: `dp=8` hosts × `8` GPUs each ⇒ 64 ranks total. |
| **5-B** | Slice out the TP sub-mesh (still 8 ranks).                                                                                                                               |
| **5-C** | Apply the same TP plan *inside each host*. Each host owns its private 1/8 weight slices.                                                                                 |
| **5-D** | Wrap the TP’ed model in FSDP so that the **whole TP shard** is further chunked across hosts. FSDP handles param shards, optim state shards, check-pointing, etc.         |

Comms hierarchy now:
‐ NVLink/PCIe (fast) handles TP collectives within a node.
‐ Infiniband/EFA (slower) handles FSDP all-gathers between nodes.

---

## 6.  Plain training loop

```python
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(1000):
    optim.zero_grad(set_to_none=True)
    loss.backward()      # TP hooks + FSDP hooks fire for you
    optim.step()
```

The loop is identical to single-GPU code.
`loss.backward()` triggers:

1. Row/col sharded FCs do reduce-scatter or all-reduce.
2. Sequence-parallel grads stay sharded.
3. FSDP unshards+reshards parameters for the full‐model all-gather.

Everything is synchronous and deterministic.

---

### Key mental picture (one last time)

```
┌────────────────────────────────────────────┐
│ 8-GPU node                                │
│  ┌── TP slice 0 ── NVLink ── TP slice 7 ┐  │
│  │  weights: 1/8 each                   │
│  │  activs: 1/8 each (seq or hidden)    │
└──┴─────────────────────────────────────────┘
          ▲ FSDP talks across hosts ▲
```

* **Tensor Parallel (TP)** = intra-node weight slicing.
* **Sequence Parallel (SP)** = intra-node activation slicing.
* **Loss Parallel** = intra-node vocab slicing during loss.
* **FSDP** = inter-node parameter sharding.

Put together, they let a 70-B-param model train on thousands of GPUs without blowing up memory or network time.

That’s every line of the six blocks, in plain language.

Correct.
Tensor-Parallel isn’t “one-click” like DDP/FSDP, and here’s why in plain words:

| Thing                | What it does                                                              | Why it can be automatic                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DDP**              | Copies whole model, sums grads.                                           | Needs **no** knowledge of math inside a layer.                                                                                                                     |
| **FSDP**             | Slices params + optim state, gathers when needed.                         | Still math-agnostic; only moves bytes around.                                                                                                                      |
| **TP / SP / Loss-P** | **Changes the math** – splits each matmul/softmax/layer-norm across GPUs. | Works **only** on ops that have a known “split rule” (Linear, Embedding, LayerNorm, etc.). Library can’t guess for an arbitrary custom layer, so you must tell it. |

### What you must specify

* **Which module names** get `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, etc.
* Optionally, input/output layouts when the default isn’t right.
* Any hyper-params that depend on the slice count (e.g. `n_heads //= tp_size`).

Anything you leave out is kept **replicated**, so a safe minimum plan is:

```python
plan = {
    name: ColwiseParallel()   # just shard the fat Linear weights
    for name, m in model.named_modules()
    if isinstance(m, nn.Linear)
}
parallelize_module(model, tp_mesh, plan)
```

…but that’s still you deciding “all Linear layers are safe to shard”.

### Can it be made automatic?

* **Template models** (Megatron-LM, HF `GPTNeoX`, etc.) ship with a helper that builds the plan for you – because the authors know every layer.
* **Research** is ongoing on graph-level auto-sharding, but nothing in PyTorch today will look at an unknown module and infer a correct TP rule.

So, yes: for now you need to know (or quickly inspect) the important sub-modules and spell out the parallel plan.

