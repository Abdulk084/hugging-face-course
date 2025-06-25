# Distributed LLM Training Interview Questions - Real World Problems

## Question 1: Setting up Distributed Data Parallel (DDP) for LLM Training

**Scenario**: You're training a 7B parameter language model across 4 GPUs. The existing code has basic model loading but lacks proper distributed setup.

**Problem**: Complete the missing distributed initialization and model wrapping code.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def setup_distributed():
    # TODO: Initialize the distributed process group
    # TODO: Set the device for current process
    pass

def create_model_and_tokenizer(model_name, device):
    # TODO: Load model and tokenizer
    # TODO: Move model to device
    # TODO: Wrap model with DDP
    pass

# Usage
if __name__ == "__main__":
    setup_distributed()
    model, tokenizer = create_model_and_tokenizer("microsoft/DialoGPT-medium", device)
```

**Solution**:
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def setup_distributed():
    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')
    
    # Set the device for current process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    return local_rank

def create_model_and_tokenizer(model_name, device):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[device])
    
    return model, tokenizer

# Usage
if __name__ == "__main__":
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    model, tokenizer = create_model_and_tokenizer("microsoft/DialoGPT-medium", device)
```

**Detailed Explanation**:
- **Process Group Initialization**: `dist.init_process_group(backend='nccl')` initializes distributed training. NCCL is optimal for GPU communication.
- **Local Rank**: Retrieved from environment variable set by the launcher (torchrun). Each process gets a unique local rank.
- **Device Setting**: `torch.cuda.set_device(local_rank)` ensures each process uses its assigned GPU.
- **DDP Wrapping**: `DistributedDataParallel` automatically handles gradient synchronization across processes.
- **Real-world Relevance**: This setup is essential for any multi-GPU LLM training, enabling efficient parallelization of large models.

---

## Question 2: Implementing Gradient Accumulation with Mixed Precision

**Scenario**: You're training a large model but memory constraints require gradient accumulation over 8 steps with mixed precision training.

**Problem**: Complete the training loop with proper gradient accumulation and mixed precision.

```python
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW

def train_step(model, dataloader, optimizer, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        # TODO: Implement gradient accumulation logic
        # TODO: Use mixed precision training
        # TODO: Handle gradient scaling
        # TODO: Synchronize gradients at the right time
        pass
    
    return total_loss / len(dataloader)
```

**Solution**:
```python
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW
import torch.distributed as dist

def train_step(model, dataloader, optimizer, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        
        # Use mixed precision training
        with autocast():
            outputs = model(input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           labels=labels)
            loss = outputs.loss / accumulation_steps  # Scale loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Accumulate gradients and update
        if (step + 1) % accumulation_steps == 0:
            # Synchronize gradients across processes
            if dist.is_initialized():
                dist.all_reduce(loss.data, op=dist.ReduceOp.SUM)
                loss.data /= dist.get_world_size()
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)
```

**Detailed Explanation**:
- **Gradient Accumulation**: Dividing loss by `accumulation_steps` ensures equivalent batch size across accumulation steps.
- **Mixed Precision**: `autocast()` automatically uses FP16 for forward pass, reducing memory usage by ~50%.
- **Gradient Scaling**: `GradScaler` prevents gradient underflow in FP16 by scaling gradients up during backward pass.
- **Distributed Synchronization**: `all_reduce` ensures all processes have the same loss value for logging purposes.
- **Memory Efficiency**: This approach allows training with effective batch sizes larger than GPU memory permits.

---

## Question 3: Implementing Model Sharding for Large Models

**Scenario**: You need to fit a 13B parameter model that doesn't fit on a single GPU using model parallelism.

**Problem**: Implement model sharding across multiple GPUs using pipeline parallelism concepts.

```python
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.distributed.pipeline.sync import Pipe

class ModelShardWrapper:
    def __init__(self, model_name, num_stages):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.num_stages = num_stages
        # TODO: Implement model sharding logic
        # TODO: Create pipeline stages
        # TODO: Handle cross-GPU communication
    
    def shard_model(self):
        # TODO: Split model layers across GPUs
        pass
    
    def create_pipeline(self):
        # TODO: Create pipeline with proper device placement
        pass
```

**Solution**:
```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.distributed.pipeline.sync import Pipe
import torch.distributed as dist

class ModelShardWrapper:
    def __init__(self, model_name, num_stages):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.num_stages = num_stages
        self.devices = [torch.device(f'cuda:{i}') for i in range(num_stages)]
        self.pipeline_model = self.create_pipeline()
    
    def shard_model(self):
        """Split transformer layers across devices"""
        layers = list(self.model.transformer.h)  # Get transformer layers
        layers_per_stage = len(layers) // self.num_stages
        
        sharded_layers = []
        for stage in range(self.num_stages):
            start_idx = stage * layers_per_stage
            end_idx = start_idx + layers_per_stage if stage < self.num_stages - 1 else len(layers)
            
            stage_layers = nn.Sequential()
            
            # Add embedding for first stage
            if stage == 0:
                stage_layers.add_module('wte', self.model.transformer.wte)
                stage_layers.add_module('wpe', self.model.transformer.wpe)
                stage_layers.add_module('drop', self.model.transformer.drop)
            
            # Add transformer layers
            for i, layer in enumerate(layers[start_idx:end_idx]):
                stage_layers.add_module(f'h_{start_idx + i}', layer)
            
            # Add final layers for last stage
            if stage == self.num_stages - 1:
                stage_layers.add_module('ln_f', self.model.transformer.ln_f)
                stage_layers.add_module('lm_head', self.model.lm_head)
            
            sharded_layers.append(stage_layers)
        
        return sharded_layers
    
    def create_pipeline(self):
        """Create pipeline with proper device placement"""
        sharded_layers = self.shard_model()
        
        # Move each stage to its designated device
        for i, stage in enumerate(sharded_layers):
            stage.to(self.devices[i])
        
        # Create pipeline
        pipeline_model = Pipe(
            nn.Sequential(*sharded_layers),
            balance=None,  # Already balanced manually
            devices=self.devices,
            chunks=1  # Number of micro-batches
        )
        
        return pipeline_model
    
    def forward(self, input_ids):
        return self.pipeline_model(input_ids)

# Usage
model_wrapper = ModelShardWrapper("gpt2-large", num_stages=4)
```

**Detailed Explanation**:
- **Layer Sharding**: Transformer layers are evenly distributed across available GPUs to balance memory usage.
- **Pipeline Creation**: `Pipe` handles automatic tensor movement between devices during forward/backward passes.
- **Device Placement**: Each stage is explicitly moved to its designated GPU before pipeline creation.
- **Memory Optimization**: Only one stage's parameters reside on each GPU, enabling larger models to fit in available memory.
- **Communication Overhead**: Pipeline parallelism trades computation efficiency for memory efficiency - suitable for very large models.

---

## Question 4: Implementing Custom Distributed Sampler for Efficient Data Loading

**Scenario**: You need to ensure each GPU processes different data batches without overlap while maintaining reproducible training.

**Problem**: Create a distributed sampler that handles uneven dataset sizes and maintains deterministic shuffling.

```python
from torch.utils.data import Sampler
import torch.distributed as dist
import torch

class DistributedSamplerCustom(Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        # TODO: Initialize distributed sampler parameters
        # TODO: Handle dataset size calculations
        # TODO: Ensure reproducible shuffling
        pass
    
    def __iter__(self):
        # TODO: Create indices for current process
        # TODO: Handle shuffling with seed
        # TODO: Ensure no data overlap between processes
        pass
    
    def __len__(self):
        # TODO: Return number of samples for current process
        pass
```

**Solution**:
```python
from torch.utils.data import Sampler
import torch.distributed as dist
import torch
import math

class DistributedSamplerCustom(Sampler):
    def __init__(self, dataset, shuffle=True, seed=0, drop_last=False):
        # Initialize distributed parameters
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Calculate samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Drop the tail of data to make it evenly divisible
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
            
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Generate indices based on epoch and seed for reproducibility
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Pad indices to make it evenly divisible
        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail samples to make it evenly divisible
            indices = indices[:self.total_size]
        
        # Subsample for current process
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        indices = indices[start_idx:end_idx]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling reproducibility"""
        self.epoch = epoch

# Usage example
from torch.utils.data import DataLoader

# Create custom sampler
sampler = DistributedSamplerCustom(dataset, shuffle=True, seed=42)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# Update epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        # Training logic
        pass
```

**Detailed Explanation**:
- **Rank and World Size**: Each process gets a unique rank, and world_size represents total number of processes.
- **Sample Distribution**: Dataset is divided evenly across processes, with padding to handle uneven divisions.
- **Reproducible Shuffling**: Using `torch.Generator` with seed ensures same shuffling order across runs.
- **Epoch-based Shuffling**: `set_epoch()` changes the random seed each epoch for different data ordering.
- **Memory Efficiency**: Each process only loads its assigned portion of data, reducing memory overhead.

---

## Question 5: Implementing Gradient Clipping and Monitoring in Distributed Training

**Scenario**: You need to implement gradient clipping that works correctly across distributed processes and monitor gradient norms.

**Problem**: Create a gradient clipping function that calculates global gradient norms across all processes.

```python
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

def distributed_gradient_clipping(model, max_norm, norm_type=2.0):
    # TODO: Calculate gradient norms across all processes
    # TODO: Perform gradient clipping based on global norm
    # TODO: Return actual gradient norm for monitoring
    pass

def log_gradient_stats(model, step):
    # TODO: Calculate and log gradient statistics
    # TODO: Handle distributed logging
    pass
```

**Solution**:
```python
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
import numpy as np

def distributed_gradient_clipping(model, max_norm, norm_type=2.0):
    """
    Perform gradient clipping with global norm calculation across distributed processes
    """
    # Get all gradients from the model
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    # Calculate local gradient norm
    device = parameters[0].grad.device
    local_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) 
                    for p in parameters]), 
        norm_type
    )
    
    # Calculate global gradient norm across all processes
    if dist.is_initialized():
        # Aggregate norms from all processes
        if norm_type == 2.0:
            # For L2 norm, sum squares then sqrt
            local_norm_squared = local_norm ** 2
            dist.all_reduce(local_norm_squared, op=dist.ReduceOp.SUM)
            global_norm = local_norm_squared ** 0.5
        else:
            # For other norms, use max reduction
            global_norm = local_norm.clone()
            dist.all_reduce(global_norm, op=dist.ReduceOp.MAX)
    else:
        global_norm = local_norm
    
    # Perform gradient clipping based on global norm
    if global_norm > max_norm:
        clip_coef = max_norm / (global_norm + 1e-6)
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return global_norm.item()

def log_gradient_stats(model, step, writer=None):
    """
    Calculate and log detailed gradient statistics
    """
    grad_norms = []
    grad_magnitudes = []
    zero_grad_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            grad_norms.append(grad_norm)
            grad_magnitudes.extend(param.grad.data.abs().flatten().tolist())
            total_params += param.numel()
        else:
            zero_grad_params += param.numel()
    
    if grad_norms:
        stats = {
            'gradient_norm_mean': np.mean(grad_norms),
            'gradient_norm_max': np.max(grad_norms),
            'gradient_norm_min': np.min(grad_norms),
            'gradient_magnitude_mean': np.mean(grad_magnitudes),
            'gradient_magnitude_std': np.std(grad_magnitudes),
            'zero_gradient_ratio': zero_grad_params / (total_params + zero_grad_params)
        }
        
        # Log only on rank 0 to avoid duplicate logs
        if not dist.is_initialized() or dist.get_rank() == 0:
            if writer:  # TensorBoard writer
                for key, value in stats.items():
                    writer.add_scalar(f'gradients/{key}', value, step)
            else:
                print(f"Step {step} - Gradient Stats: {stats}")
        
        return stats
    
    return None

# Usage in training loop
def training_step(model, optimizer, loss, step, max_grad_norm=1.0):
    # Backward pass
    loss.backward()
    
    # Log gradient statistics before clipping
    grad_stats = log_gradient_stats(model, step)
    
    # Perform distributed gradient clipping
    grad_norm = distributed_gradient_clipping(model, max_grad_norm)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Log gradient norm after clipping
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Step {step}: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}")
    
    return grad_norm, grad_stats
```

**Detailed Explanation**:
- **Global Norm Calculation**: For L2 norm, squares are summed across processes then square-rooted to get true global norm.
- **Distributed Clipping**: All processes use the same global norm for clipping, ensuring consistent gradients.
- **Gradient Monitoring**: Tracks various statistics including norm distribution and zero-gradient ratios.
- **Numerical Stability**: Small epsilon prevents division by zero in clipping coefficient calculation.
- **Rank-based Logging**: Only rank 0 logs to prevent duplicate entries in distributed training.

---

## Question 6: Implementing Dynamic Loss Scaling for Mixed Precision Training

**Scenario**: You need to implement adaptive loss scaling that adjusts based on gradient overflow detection.

**Problem**: Create a dynamic loss scaler that increases/decreases scale based on training stability.

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=1000):
        # TODO: Initialize scaling parameters
        # TODO: Track overflow history
        pass
    
    def scale_loss(self, loss):
        # TODO: Scale loss for backward pass
        pass
    
    def unscale_gradients(self, optimizer):
        # TODO: Unscale gradients and detect overflow
        pass
    
    def update_scale(self, found_inf):
        # TODO: Update scale based on overflow detection
        pass
```

**Solution**:
```python
import torch
import torch.distributed as dist
from collections import defaultdict

class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=1000, min_scale=1.0):
        """
        Dynamic loss scaler for mixed precision training
        
        Args:
            init_scale: Initial loss scale value
            scale_factor: Factor to multiply/divide scale by
            scale_window: Number of steps without overflow before increasing scale
            min_scale: Minimum allowed scale value
        """
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        
        # Tracking variables
        self.growth_tracker = 0
        self.overflow_tracker = defaultdict(int)
        self.last_overflow_step = -1
        self.current_step = 0
        
    def scale_loss(self, loss):
        """Scale loss for backward pass"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """
        Unscale gradients and detect overflow
        Returns: Boolean indicating if overflow was found
        """
        found_inf = False
        
        # Unscale gradients for each parameter group
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Check for inf/nan before unscaling
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        found_inf = True
                        break
                    
                    # Unscale gradient
                    param.grad.data.div_(self.scale)
                    
                    # Check for inf/nan after unscaling
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        found_inf = True
                        break
            
            if found_inf:
                break
        
        # In distributed training, check if any process found overflow
        if dist.is_initialized():
            # Convert to tensor for all_reduce
            found_inf_tensor = torch.tensor(found_inf, dtype=torch.bool, device='cuda')
            dist.all_reduce(found_inf_tensor, op=dist.ReduceOp.MAX)
            found_inf = found_inf_tensor.item()
        
        return found_inf
    
    def update_scale(self, found_inf):
        """Update scale based on overflow detection"""
        self.current_step += 1
        
        if found_inf:
            # Overflow detected - reduce scale and reset growth tracker
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self.growth_tracker = 0
            self.last_overflow_step = self.current_step
            self.overflow_tracker[self.current_step] = 1
            
            # Zero out gradients to prevent corrupted updates
            return False  # Skip optimizer step
        else:
            # No overflow - increment growth tracker
            self.growth_tracker += 1
            
            # Increase scale if we've had enough successful steps
            if self.growth_tracker >= self.scale_window:
                self.scale *= self.scale_factor
                self.growth_tracker = 0
            
            return True  # Proceed with optimizer step
    
    def get_scale(self):
        """Get current scale value"""
        return self.scale
    
    def get_stats(self):
        """Get scaling statistics for monitoring"""
        recent_overflows = sum(1 for step in self.overflow_tracker.keys() 
                             if self.current_step - step <= 1000)
        
        return {
            'loss_scale': self.scale,
            'growth_tracker': self.growth_tracker,
            'steps_since_overflow': self.current_step - self.last_overflow_step,
            'recent_overflows': recent_overflows,
            'total_overflows': len(self.overflow_tracker)
        }

# Usage in training loop
def train_with_dynamic_scaling(model, dataloader, optimizer):
    scaler = DynamicLossScaler(init_scale=2**16, scale_window=1000)
    
    for step, batch in enumerate(dataloader):
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        # Scale loss and backward pass
        scaled_loss = scaler.scale_loss(loss)
        scaled_loss.backward()
        
        # Unscale gradients and check for overflow
        found_inf = scaler.unscale_gradients(optimizer)
        
        # Update scale and conditionally step optimizer
        should_step = scaler.update_scale(found_inf)
        
        if should_step:
            # Apply gradient clipping on unscaled gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Log statistics periodically
        if step % 100 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            stats = scaler.get_stats()
            print(f"Step {step}: Loss={loss.item():.4f}, Scale={stats['loss_scale']:.0f}, "
                  f"Overflows={stats['recent_overflows']}")
```

**Detailed Explanation**:
- **Adaptive Scaling**: Scale increases after successful steps without overflow, decreases when overflow detected.
- **Overflow Detection**: Checks for inf/nan values both before and after unscaling gradients.
- **Distributed Coordination**: All processes must agree on overflow detection using `all_reduce` with MAX operation.
- **Gradient Zeroing**: When overflow occurs, gradients are implicitly zeroed by skipping optimizer step.
- **Statistics Tracking**: Monitors overflow frequency and scaling behavior for debugging and optimization.

---

## Question 7: Implementing Checkpointing with Distributed State Management

**Scenario**: You need to save and load model checkpoints that include distributed training state across multiple processes.

**Problem**: Create a checkpointing system that handles distributed model states, optimizer states, and training metadata.

```python
import torch
import torch.distributed as dist
import os
from pathlib import Path

class DistributedCheckpointer:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None):
        # TODO: Initialize checkpointing system
        # TODO: Handle distributed state collection
        pass
    
    def save_checkpoint(self, epoch, step, loss, metrics=None):
        # TODO: Collect states from all processes
        # TODO: Save checkpoint with proper naming
        # TODO: Handle cleanup of old checkpoints
        pass
    
    def load_checkpoint(self, checkpoint_path=None):
        # TODO: Load checkpoint and distribute states
        # TODO: Handle missing checkpoint gracefully
        # TODO: Return loaded metadata
        pass
    
    def get_latest_checkpoint(self):
        # TODO: Find most recent checkpoint
        pass
```

**Solution**:
```python
import torch
import torch.distributed as dist
import os
import json
import glob
from pathlib import Path
from datetime import datetime

class DistributedCheckpointer:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None, max_checkpoints=5):
        """
        Distributed checkpointing system
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model: Model to checkpoint (should be DDP wrapped)
            optimizer: Optimizer state to save
            scheduler: Optional learning rate scheduler
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_checkpoints = max_checkpoints
        
        # Get distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        self.is_main_process = self.rank == 0
    
    def save_checkpoint(self, epoch, step, loss, metrics=None, save_optimizer=True):
        """
        Save distributed checkpoint
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            metrics: Optional metrics dictionary
            save_optimizer: Whether to save optimizer state
        """
        # Only main process handles file I/O coordination
        if self.is_main_process:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            checkpoint_path.mkdir(exist_ok=True)
        else:
            checkpoint_path = None
        
        # Broadcast checkpoint path to all processes
        if dist.is_initialized():
            # Convert path to string for broadcasting
            if self.is_main_process:
                path_str = str(checkpoint_path)
            else:
                path_str = ""
            
            # Broadcast path length first, then path
            path_tensor = torch.zeros(256, dtype=torch.uint8, device='cuda')
            if self.is_main_process:
                path_bytes = path_str.encode('utf-8')
                path_tensor[:len(path_bytes)] = torch.frombuffer(path_bytes, dtype=torch.uint8)
            
            dist.broadcast(path_tensor, src=0)
            
            # Decode path on all processes
            path_bytes = path_tensor.cpu().numpy().tobytes().rstrip(b'\x00')
            checkpoint_path = Path(path_bytes.decode('utf-8'))
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'world_size': self.world_size,
        }
        
        # Save model state (extract from DDP wrapper if needed)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Save on main process only
        if self.is_main_process:
            # Save model state
            torch.save(model_state, checkpoint_path / 'model.pt')