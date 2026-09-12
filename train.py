import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import BPETokenizer
from dataset import TelemetryDataset
from model import TelemetryGPT
import time
import os

import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)



def split_dataset(dataset,train_ratio=0.9):
    train_size=int(len(dataset) * train_ratio)
    val_size=len(dataset) - train_size
    return torch.utils.data.random_split(dataset,[train_size,val_size])

@torch.no_grad()
def evaluate(model,val_loader,device,num_batches=20):
    #Validation loss hesapla
    model.eval()
    total_loss=0
    count=0

    for x , y in val_loader:
        if count >= num_batches:
            break

        x,y=x.to(device) , y.to(device)
        _,loss =model(x,y)
        total_loss += loss.item()
        count +=1

    model.train()
    return total_loss / count

def save_checkpoints(model,optimizer,epoch,step , loss,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save({
        "Epoch":epoch,
        "Step":step,
        "Loss":loss,
        "Model":model.state_dict(),
        "Optimizer":optimizer.state_dict()
    },path)
    print(f"Checkpoint kaydedildi:{path}")



def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Checkpoint yüklendi: {path}")
    return checkpoint["epoch"], checkpoint["step"]


def train ():
    #1.Device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:{device}")

    #2.Tokenizer + Dataset
    print("\nTokenizer yükleniyor..")
    tokenizer=BPETokenizer.load("tokenizer.json")

    print("\nDataset yükleniyor..")
    dataset=TelemetryDataset("data/telemetry_dataset.jsonl",tokenizer,max_length=config["block_size"])
    train_set,val_set=split_dataset(dataset,config["train_split"])

    train_loader=DataLoader(train_set,config["batch_size"],shuffle=True)
    val_loader=DataLoader(val_set,config["batch_size"],shuffle=False)

    print(f"Train:{len(train_set)} Validation:{len(val_set)}")

    #3.Model
    print("Model oluşturuluyor..")
    model=TelemetryGPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    ).to(device)

    #4.Optimizer
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])

    # 5. Learning Rate Scheduler (opsiyonel ama faydalı)
    # Eğitim boyunca lr'yi yavaşça düşürür
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] * len(train_loader)
    )

    print("\nEğitim başlıyor ..\n")
    global_step=0

    for epoch in range(config["epochs"]):
        epoch_start=time.time()
        epoch_loss=0

        for step , (x,y) in enumerate(train_loader):
            x, y = x.to(device) , y.to(device)

            #forward
            logits,loss= model(x,y)

            #Backward
            optimizer.zero_grad()
            loss.backward()

            #Graddient clipping -> patlamayı önler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step +=1


            #Log
            if global_step % 50 ==0:
                avg_loss=epoch_loss / (step +1)
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}/{config['epochs']} | "
                      f"Step {global_step} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.6f}")

            #Validation
            if global_step % config["eval_interval"] == 0:
                val_loss=evaluate(model,val_loader,device)
                print(f"\n Val Loss: {val_loss:.4f}\n")

            #Checkpoint
            if global_step % config["save_interval"] == 0:
                path = f"{config['checkpoint_dir']}/step_{global_step}.pt"
                save_checkpoints(model, optimizer, epoch, global_step, loss.item(), path)

        # Epoch sonu
        epoch_time = time.time() - epoch_start
        print(f"\n✅ Epoch {epoch + 1} tamamlandı | "
              f"Avg Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Süre: {epoch_time:.1f}s\n")

    # 7. Final model kaydet
    save_checkpoints(model, optimizer, config["epochs"], global_step, loss.item(),
                    f"{config['checkpoint_dir']}/final.pt")
    print("\n Eğitim tamamlandı!")



if __name__ == "__main__":
    train()
