# Distributed LLM Training Cheat Sheet - Interview Ready

## 🚀 Quick Search Index
- [Basic Distributed Setup](#basic-distributed-setup)
- [DDP (Distributed Data Parallel)](#ddp-distributed-data-parallel)
- [Mixed Precision Training](#mixed-precision-training)
- [Gradient Accumulation](#gradient-accumulation)
- [Pipeline Parallelism](#pipeline-parallelism)
- [Loss Synchronization](#loss-synchronization)
- [Checkpointing](#checkpointing)
- [Data Loading](#data-loading)
- [Common Patterns](#common-patterns)
- [Debugging & Monitoring](#debugging--monitoring)

---

## Basic Distributed Setup

### Initialize Distributed Process Group
```python
import torch.distributed as dist
import os

# Basic initialization
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# With error handling
def setup_distributed_safe():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1
```

### Environment Variables Check
```python
# Check if distributed is available
def is_distributed():
    return dist.is_available() and dist.is_initialized()

# Get distributed info
def get_dist_info():
    if is_distributed():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1
```

---

## DDP (Distributed Data Parallel)

### Basic DDP Setup
```python
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel

def create_ddp_model(model_name, device):
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model

# Find unused parameters (for complex models)
def create_ddp_model_safe(model_name, device):
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    return model
```

### DDP with Gradient Synchronization Control
```python
# Manual gradient synchronization
def forward_with_manual_sync(model, batch, no_sync=False):
    if no_sync and isinstance(model, DDP):
        with model.no_sync():
            outputs = model(**batch)
            loss = outputs.loss
            return loss
    else:
        outputs = model(**batch)
        loss = outputs.loss
        return loss
```

---

## Mixed Precision Training

### Basic Mixed Precision
```python
from torch.cuda.amp import GradScaler, autocast

# Setup
scaler = GradScaler()

# Forward pass with autocast
def forward_with_amp(model, batch):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    return loss

# Training step
def amp_training_step(model, batch, optimizer, scaler):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss
```

### Dynamic Loss Scaling
```python
def training_step_with_dynamic_scaling(model, batch, optimizer, scaler):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Scale and backward
    scaler.scale(loss).backward()
    
    # Unscale for gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step and update
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    return loss
```

---

## Gradient Accumulation

### Basic Gradient Accumulation
```python
def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps):
    model.train()
    
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return loss
```

### Gradient Accumulation + Mixed Precision
```python
def train_with_grad_accum_amp(model, dataloader, optimizer, scaler, accumulation_steps):
    model.train()
    
    for step, batch in enumerate(dataloader):
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    return loss
```

### Gradient Accumulation + DDP
```python
def train_ddp_grad_accum(model, dataloader, optimizer, accumulation_steps):
    model.train()
    
    for step, batch in enumerate(dataloader):
        # Disable sync for accumulation steps
        with model.no_sync() if (step + 1) % accumulation_steps != 0 else nullcontext():
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return loss
```

---

## Pipeline Parallelism

### Basic Pipeline Setup
```python
from torch.distributed.pipeline.sync import Pipe
import torch.nn as nn

def create_pipeline_model(model, num_stages, devices):
    # Split model into stages
    layers = list(model.children())
    layers_per_stage = len(layers) // num_stages
    
    stages = []
    for i in range(num_stages):
        start = i * layers_per_stage
        end = start + layers_per_stage if i < num_stages - 1 else len(layers)
        stage = nn.Sequential(*layers[start:end]).to(devices[i])
        stages.append(stage)
    
    pipeline = Pipe(nn.Sequential(*stages), balance=None, devices=devices, chunks=4)
    return pipeline
```

### Transformer Pipeline Sharding
```python
def shard_transformer_pipeline(model, num_stages):
    # Get transformer layers
    layers = list(model.transformer.h)
    layers_per_stage = len(layers) // num_stages
    
    sharded_stages = []
    for stage in range(num_stages):
        start_idx = stage * layers_per_stage
        end_idx = start_idx + layers_per_stage if stage < num_stages - 1 else len(layers)
        
        stage_layers = nn.Sequential()
        
        # First stage: embeddings
        if stage == 0:
            stage_layers.add_module('wte', model.transformer.wte)
            stage_layers.add_module('wpe', model.transformer.wpe)
            stage_layers.add_module('drop', model.transformer.drop)
        
        # Add transformer layers
        for i, layer in enumerate(layers[start_idx:end_idx]):
            stage_layers.add_module(f'h_{start_idx + i}', layer)
        
        # Last stage: output layers
        if stage == num_stages - 1:
            stage_layers.add_module('ln_f', model.transformer.ln_f)
            stage_layers.add_module('lm_head', model.lm_head)
        
        sharded_stages.append(stage_layers)
    
    return sharded_stages
```

---

## Loss Synchronization

### Basic Loss Sync
```python
def synchronize_loss(loss):
    if dist.is_initialized():
        dist.all_reduce(loss.data, op=dist.ReduceOp.SUM)
        loss.data /= dist.get_world_size()
    return loss

# Using detach (preferred)
def synchronize_loss_detached(loss):
    if dist.is_initialized():
        sync_loss = loss.detach()
        dist.all_reduce(sync_loss, op=dist.ReduceOp.SUM)
        sync_loss /= dist.get_world_size()
        return sync_loss
    return loss
```

### Sample-Weighted Loss Sync
```python
def synchronize_weighted_loss(loss, batch_size):
    if dist.is_initialized():
        # Weight by sample count
        weighted_loss = loss.detach() * batch_size
        total_samples = torch.tensor(batch_size, dtype=torch.float, device=loss.device)
        
        dist.all_reduce(weighted_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        
        return weighted_loss / total_samples
    return loss
```

---

## Checkpointing

### Basic Distributed Checkpointing
```python
def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir):
    if dist.get_rank() == 0:  # Only rank 0 saves
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'world_size': dist.get_world_size()
        }
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    return None

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle DDP wrapper
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']
```

### Checkpoint with State Synchronization
```python
def save_checkpoint_sync(model, optimizer, epoch, step, loss, checkpoint_dir):
    # Ensure all processes reach this point
    if dist.is_initialized():
        dist.barrier()
    
    if dist.get_rank() == 0:
        # Extract model state (handle DDP)
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'world_size': dist.get_world_size(),
            'rank': dist.get_rank()
        }
        
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    # Wait for save to complete
    if dist.is_initialized():
        dist.barrier()
```

---

## Data Loading

### Distributed Sampler
```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def create_distributed_dataloader(dataset, batch_size, shuffle=True):
    sampler = DistributedSampler(
        dataset, 
        shuffle=shuffle,
        drop_last=True
    ) if dist.is_initialized() else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, sampler
```

### Custom Distributed Sampler
```python
from torch.utils.data import Sampler
import math

class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Pad to make evenly divisible
        indices += indices[:self.total_size - len(indices)]
        
        # Subsample for current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
```

---

## Common Patterns

### Complete Training Step
```python
def complete_training_step(model, batch, optimizer, scaler, accumulation_steps, max_grad_norm=1.0):
    # Move batch to device
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Forward pass with mixed precision
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient accumulation check
    if (step + 1) % accumulation_steps == 0:
        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        return loss * accumulation_steps, grad_norm
    
    return loss * accumulation_steps, None
```

### Model + Optimizer Setup
```python
def setup_model_optimizer(model_name, learning_rate, device):
    from transformers import AutoModelForCausalLM, AdamW
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    
    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[device])
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    return model, optimizer, scaler
```

### Learning Rate Scheduler
```python
def setup_scheduler(optimizer, num_training_steps, warmup_steps=0):
    from transformers import get_linear_schedule_with_warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler
```

---

## Debugging & Monitoring

### Memory Monitoring
```python
def log_memory_usage(device=None):
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    
    print(f"GPU {device}: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB")
```

### Gradient Monitoring
```python
def log_gradient_norms(model):
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm:.4f}, Params with grad: {param_count}")
    return total_norm
```

### Distributed Debugging
```python
def debug_distributed_setup():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}, Device: cuda:{local_rank}")
        
        # Test communication
        tensor = torch.tensor([rank], dtype=torch.float).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(world_size))
        
        print(f"Communication test: {tensor.item()} == {expected_sum}: {tensor.item() == expected_sum}")
    else:
        print("Distributed not initialized")
```

### Performance Profiling
```python
def profile_training_step(model, batch, optimizer, scaler):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Quick Fixes & Common Issues

### OOM (Out of Memory) Fixes
```python
# Clear cache
torch.cuda.empty_cache()

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(model_layer, input_tensor)

# Reduce batch size dynamically
def adaptive_batch_size(initial_batch_size, model, sample_batch):
    batch_size = initial_batch_size
    while batch_size > 0:
        try:
            test_batch = {k: v[:batch_size] for k, v in sample_batch.items()}
            _ = model(**test_batch)
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise
    return 1
```

### Loss Scaling Issues
```python
# Check for inf/nan in loss
def safe_loss_backward(loss, scaler):
    if torch.isfinite(loss):
        scaler.scale(loss).backward()
        return True
    else:
        print(f"Warning: Non-finite loss detected: {loss.item()}")
        return False
```

### Gradient Explosion
```python
# Aggressive gradient clipping
def safe_gradient_clip(model, max_norm=0.5):
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if grad_norm > max_norm * 2:
        print(f"Warning: Large gradient norm: {grad_norm:.4f}")
    return grad_norm
```

---

## Launch Commands (for reference)

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train_script.py

# Multiple nodes
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 train_script.py

# Environment variables to check
echo $LOCAL_RANK $RANK $WORLD_SIZE $MASTER_ADDR $MASTER_PORT
```