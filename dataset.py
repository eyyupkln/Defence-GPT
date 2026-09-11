import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer


class TelemetryDataset(Dataset) :
    def __init__(self,json_path,tokenizer,max_length=128):
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.data=[]

        print("Veri yükleniyor..")
        with open(json_path, "r", encoding="utf-8") as f :
            for line in f :
                entry=json.loads(line)
                self.data.append(entry["full_text"])
            print(f"{len(self.data)} örnek yüklendi.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text=self.data[item]

        #Encode
        token_ids=self.tokenizer.encode(text)
        pad_id = self.tokenizer.special_tokens["<PAD>"]
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [pad_id] * padding_length

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Decoder-only için:
        # input  → [t0, t1, t2, ..., t_n-1]  (son token hariç)
        # target → [t1, t2, t3, ..., t_n]     (ilk token hariç)
        # Model her adımda bir sonraki tokeni tahmin eder
        x = token_ids[:-1]  # input
        y = token_ids[1:]  # target (bir kaydırılmış)

        return x, y


#---Test----
if __name__ == "__main__":
    tokenizer=BPETokenizer.load("tokenizer.json")
    dataset=TelemetryDataset("data/telemetry_dataset.jsonl",tokenizer,max_length=128)

    #Dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # İlk batch'e bak
    x, y = next(iter(dataloader))
    print(f"Input shape  : {x.shape}")  # [32, 127]
    print(f"Target shape : {y.shape}")  # [32, 127]
    print(f"\nİlk örnek input  : {x[0][:10]}...")
    print(f"İlk örnek target : {y[0][:10]}...")

    # Decode ederek kontrol et
    print(f"\nDecode input  : {tokenizer.decode(x[0].tolist())[:80]}...")
    print(f"Decode target : {tokenizer.decode(y[0].tolist())[:80]}...")