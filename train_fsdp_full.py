import os 
import sys
import torch
import contextlib
import torch.distributed as dist
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding)
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset
from torch.amp import autocast, GradScaler
from torch.optim import AdamW, lr_scheduler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType


def ddp_setup(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    dist.destroy_process_group()


def build_data_loaders(rank: int, world_size: int, tokenizer, batch_size: int,
                       dataset_name: str):
    
    def tokenization(example, column):
        return tokenizer(example[column], truncation=True, max_length=128)
    
    ds_raw = load_dataset(dataset_name)
    column = "text" if "text" in ds_raw["train"].column_names else ds_raw["train"].column_names[0]
    ds_tk = ds_raw.map(tokenization, fn_kwargs={"column": column}, batched=True, remove_columns=[column])

    ds_tk.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    sampler_train = DistributedSampler(ds_tk["train"], num_replicas=world_size,
                                       rank=rank, shuffle=True, drop_last=True)
    sampler_test = DistributedSampler(ds_tk["test"], num_replicas=world_size,
                                      rank=rank, shuffle=False, drop_last=False)
    
    dl_train = DataLoader(dataset=ds_tk["train"], sampler=sampler_train,
                         batch_size=batch_size,
                         collate_fn=collator,
                         pin_memory=True)
    dl_test = DataLoader(dataset=ds_tk["test"], sampler=sampler_test,
                        batch_size=batch_size,
                        collate_fn=collator,
                        pin_memory=True)
    labels_names_list = ds_tk["train"].features["label"].names

    return dl_train, dl_test, sampler_train, len(labels_names_list)


@torch.no_grad()
def evaluate(model, dl_test, rank, world_size):
    model.eval()

    correct_total = torch.tensor(0, device=rank)
    loss_sum = torch.tensor(0.0, device=rank)
    total = torch.tensor(0, device=rank)

    for step, batch in enumerate(dl_test):
        inp = {k: v.cuda(rank) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].cuda(rank)
        
        with autocast("cuda"):
            output = model(**inp, labels=labels)
        loss = output.loss * len(labels)
        loss_sum = loss_sum + loss

        predictions = output.logits.argmax(axis=-1)
        correct = (predictions == labels).sum()
        correct_total = correct_total + correct
        total = total + len(labels)

    dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"loss is {(loss_sum.item()/total.item()):.6f} | "
              f"accuracy is {(correct_total.item()/total.item()):.3f}")

    model.train()


def save_checkpoint(model, optimizer, scaler, global_step, epoch, filepath, rank):
    """Save FSDP checkpoint - only rank 0 saves"""
    if rank == 0:
        # Configure to gather full state dict on CPU, rank 0 only
        fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        # Get the full unsharded state dict
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            full_state_dict = model.state_dict()
        
        checkpoint = {
            "model": full_state_dict,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at global step {global_step}, epoch {epoch}")


def load_checkpoint(model, optimizer, scaler, checkpoint_path, rank):
    """Load FSDP checkpoint"""
    # Load checkpoint on CPU first
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Configure for loading full state dict
    fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    # Load model state dict using FSDP context
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
        model.load_state_dict(checkpoint["model"])
    
    # Load optimizer and scaler states
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    
    global_step = checkpoint["global_step"]
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
    
    if rank == 0:
        print(f"Loaded checkpoint from {checkpoint_path}: global_step={global_step}, start_epoch={start_epoch}")
    
    return global_step, start_epoch


def main_train_fsdp(rank: int, world_size: int, batch_size: int,
                    accum: int, base_lr: float, 
                    num_epochs: int, model_name: str, 
                    dataset_name: str, resume_path: str | None = None):
    
    torch.manual_seed(42 + rank)
    ddp_setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dl_train, dl_test, sampler_train, num_labels = build_data_loaders(
        rank=rank, world_size=world_size, tokenizer=tokenizer,
        batch_size=batch_size, dataset_name=dataset_name)
    
    # Model starts on CPU for FSDP
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    model = FSDP(model, 
                auto_wrap_policy=default_auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=True),
                device_id=rank)

    opt = AdamW(model.parameters(), lr=base_lr * world_size)
    schd = lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.8)
    scaler = GradScaler("cuda")

    global_step = 0
    start_epoch = 0

    # Load checkpoint if resuming
    if resume_path and os.path.exists(resume_path):
        global_step, start_epoch = load_checkpoint(model, opt, scaler, resume_path, rank)
    
    # Broadcast counters to ensure all ranks are synchronized
    global_step_t = torch.tensor([global_step], device=rank)
    start_epoch_t = torch.tensor([start_epoch], device=rank)
    
    dist.broadcast(global_step_t, src=0)
    dist.broadcast(start_epoch_t, src=0)
    
    global_step = int(global_step_t.item())
    start_epoch = int(start_epoch_t.item())

    for epoch in range(start_epoch, num_epochs):
        sampler_train.set_epoch(epoch)

        for step, batch in enumerate(dl_train):
            inp = {k: v.cuda(rank) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].cuda(rank)

            if step % accum == 0:
                opt.zero_grad(set_to_none=True)
            
            # Control gradient synchronization
            
            
            
            with autocast("cuda"):
                output = model(**inp, labels=labels)
            loss = output.loss * (1 / accum)
            scaler.scale(loss).backward()

            # Update on last accumulation step
            if step % accum == (accum - 1):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                # Calculate and reduce loss for logging
                detached_loss = loss.detach() * accum
                dist.all_reduce(detached_loss, op=dist.ReduceOp.AVG)

                if rank == 0 and global_step % 100 == 0:  # Log every 100 steps
                    print(f"Global step {global_step}, epoch {epoch}, avg loss: {detached_loss.item():.6f}")

                # Save checkpoint periodically
                if rank == 0 and global_step % 1000 == 0:  # Save every 1000 steps
                    save_checkpoint(model, opt, scaler, global_step, epoch, "model.ckpt", rank)
                
                dist.barrier()
                global_step += 1

        # Evaluate at end of each epoch
        evaluate(model, dl_test, rank, world_size)
        schd.step()
        
        # Save checkpoint at end of each epoch
        if rank == 0:
            save_checkpoint(model, opt, scaler, global_step, epoch, f"model_epoch_{epoch}.ckpt", rank)

    # Final checkpoint save
    if rank == 0:
        save_checkpoint(model, opt, scaler, global_step, num_epochs-1, "model_final.ckpt", rank)
        print("Training finished")

    ddp_cleanup()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    accum = int(os.getenv("ACCUM", "4"))
    base_lr = float(os.getenv("BASE_LR", "1e-4"))
    num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
    model_name = os.getenv("MODEL_NAME", "bert-base-uncased")
    dataset_name = os.getenv("DATASET_NAME", "ag_news")
    resume_path = os.getenv("RESUME")

    main_train_fsdp(rank=rank, 
                   world_size=world_size,
                   batch_size=batch_size,
                   accum=accum,
                   base_lr=base_lr,
                   num_epochs=num_epochs,
                   model_name=model_name,
                   dataset_name=dataset_name,
                   resume_path=resume_path)