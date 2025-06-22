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



def ddp_setup(rank:int, world_size:int)->None:

    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    dist.destroy_process_group()

def build_data_loaders(rank:int, world_size:int, tokenizer, batch_size:int,
                       dataset_name:str):
    
    def tokenization(example, column):
        return tokenizer(example[column], truncation = True, max_length = 128)
    

    ds_raw = load_dataset(dataset_name)
    column = "text" if "text" in ds_raw["train"].column_names else ds_raw["train"].column_names[0]
    ds_tk = ds_raw.map(tokenization, fn_kwargs = {"column" : column},  batched=True, remove_columns=[column])

    ds_tk.set_format(type="torch")
    #ds_tk["train"] = ds_tk["train"].select(range(1000))

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt") # this takes tokenizer and not dataset


    sampler_train = DistributedSampler(ds_tk["train"], num_replicas=world_size,
                                       rank=rank, shuffle=True, drop_last=True )
    sampler_test = DistributedSampler(ds_tk["test"], num_replicas=world_size,
                                       rank=rank, shuffle=False, drop_last= False)
    
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
    loss_sum = torch.tensor(0, device=rank)
    total = torch.tensor(0, device = rank)



    for step, batch in enumerate(dl_test):
        inp = {k: v.cuda(rank) for k, v in batch.items() if k!="labels"}
        labels = batch["labels"].cuda(rank)
        
        with autocast("cuda"):
            output = model(**inp, labels = labels)
        loss = output.loss # this gives average loss per sample, so for total we need to 
        # calculate the loss per batch
        loss = loss * len(labels)
        loss_sum = loss_sum+loss

        predictions = output.logits.argmax(axis=-1)
        correct = (predictions ==labels).sum() # per current batch correct
        correct_total = correct_total + correct #accumulative correct
        total = total+len(labels)

    # when we are out of loop then we have three main values for each rank
    # we need to take sum of all across all ranks
    dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    
    if rank == 0:
        print(f"loss is  {(loss_sum.item()/total.item()):.6f} | "
              f"accuracy is {(correct_total.item()/total.item()):.3f}")

    model.train()


def main_train_dpp(rank:int, world_size: int, batch_size: int,
                   accum:int, base_lr : float, 
                   num_epochs: int, model_name:str, 
                   dataset_name :str, resume_path: str | None = None):
    
    torch.manual_seed(42+rank) # each rank gets seed
    ddp_setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dl_train, dl_test, sampler_train, num_labels = build_data_loaders(rank=rank,
                                                                        world_size=world_size,
                                                                        tokenizer=tokenizer,
                                                                        batch_size=batch_size,
                                                                        dataset_name = dataset_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])


    
    opt = AdamW(model.parameters(), lr = base_lr*world_size)
    schd = lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.8)
    scaler = GradScaler("cuda")

    global_step = 0
    start_epoch = 0

    if resume_path:
        ckpt = torch.load(resume_path, map_location=f"cuda:{rank}")
        model.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt['scaler'])
        global_step = ckpt['global_step']+1
        start_epoch = ckpt['start_epoch']+1

        if rank==0:
            print(f"this is [rank 0] loading from path {resume_path} resuming from global step {global_step} and epoch {start_epoch}")
    
    # broadcast some counter as well

    #global_step_t = torch.tensor([global_step]).cuda(rank) 
    #start_epoch_t = torch.tensor([start_epoch]).cuda(rank)
    # above creates tensors first on cpu and then takes to gpu rank

    global_step_t = torch.tensor([global_step], device=rank)
    start_epoch_t = torch.tensor([start_epoch], device=rank)
    # above is directly creating on gpu

    dist.broadcast(global_step_t, src=0)
    dist.broadcast(start_epoch_t, src=0)
    # broadcast needs src
    global_step = int(global_step_t.item())
    start_epoch = int(start_epoch_t.item())

    for epoch in range(start_epoch, num_epochs):
        sampler_train.set_epoch(epoch)

        for step, batch in enumerate(dl_train):

            inp ={k:v.cuda(rank, ) for k, v in batch.items() if k !="labels"}
            labels = batch["labels"].cuda(rank)

            if step% accum ==0:
                opt.zero_grad(set_to_none = True) # if its the start, start accumulating gradients
            ctx = model.no_sync() if step%accum<(accum-1) else contextlib.nullcontext()
            with ctx:
                with autocast("cuda"):
                    output = model(**inp, labels = labels)
                loss = output.loss*(1/accum)
                scaler.scale(loss).backward()

            if step%accum==(accum-1):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                detached_loss = loss.detach()*accum
                dist.all_reduce(detached_loss, op=dist.ReduceOp.AVG)

                if rank == 0 and global_step % 1 ==0:
                    print(f"with global_setp {global_step} and epoch {epoch}, the average loss is {detached_loss.item()}:.6f")
                    torch.save({"model" :model.module.state_dict(),
                                "opt": opt.state_dict(),
                                "scaler": scaler.state_dict(),
                                "global_step": global_step,
                                "epoch" :epoch}, "model.ckpt")
                dist.barrier() # every process should call it.
                global_step = global_step+1
        #evaluate model just before the new epoch starts
        evaluate(model, dl_test, rank, world_size)
        schd.step()
    ddp_cleanup()

    # this is gonna be the last message printed
    if rank == 0:
        print("training finished")

if __name__=="__main__":

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    accum = int(os.getenv("ACCUM", "4"))
    base_lr = float(os.getenv("BASE_LR" , "1e-4"))
    num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
    model_name = os.getenv("MODEL_NAME", "bert-base-uncased")
    dataset_name = os.getenv("DATASET_NAME", "ag_news")
    resume_path = os.getenv("RESUME")

    main_train_dpp(rank = rank, 
                   world_size = world_size,
                   batch_size= batch_size,
                   accum= accum,
                   base_lr=  base_lr,
                   num_epochs= num_epochs,
                   model_name = model_name,
                   dataset_name = dataset_name,
                   resume_path = resume_path)

