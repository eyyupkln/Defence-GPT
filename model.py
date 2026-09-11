import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.ln1=nn.LayerNorm(embedding_dim)

        self.attention=nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2=nn.LayerNorm(embedding_dim)

        self.mlp=nn.Sequential(
            nn.Linear(embedding_dim,4 *embedding_dim),
            nn.GELU(),        #Endüstride popüler.
            nn.Linear(embedding_dim *4 ,embedding_dim),
            nn.Dropout(dropout)
        )

    def forward (self,x,causal_mask):
        # Attention + residual
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=False
        )
        x = x + attn_output

        # FeedForward + residual
        x = x + self.mlp(self.ln2(x))

        return x



class TelemetryGPT(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 num_heads=4,
                 num_layers=4,
                 block_size=128,
                 dropout=0.1):
        super().__init__()

        self.block_size=block_size

        #Token + position embedding
        self.token_embedding= nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding=nn.Embedding(block_size,embedding_dim)
        self.dropout=nn.Dropout(dropout)

        #Transformer blokları
        self.blocks=nn.ModuleList([
            TransformerBlock(embedding_dim,num_heads,dropout)
            for _ in range(num_layers)
        ])

        #Final Layernorm + Output
        self.ln_final=nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        #Loss
        self.loss_fn=nn.CrossEntropyLoss(ignore_index=0)  # 0 = <PAD>

        # Causal mask (register_buffer → GPU'ya otomatik taşınır)
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        # Weight init
        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())

        print(f"TelemetryGPT oluşturuldu: {total:,} parametre")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,input_ids,targets=None):

        batch_size,seq_len=input_ids.shape
        device=input_ids.device

        #Embedding
        token_emb=self.token_embedding(input_ids)
        pos_emb=self.position_embedding(
            torch.arange(seq_len, device=device)
        )

        x=self.dropout(token_emb + pos_emb)

        # Causal mask slice
        mask = self.causal_mask[:seq_len, :seq_len]

        #Transformer blokları
        for block in self.blocks:
            x = block(x, mask)

        #Output
        x=self.ln_final(x)
        logits=self.output_proj(x)

        # Loss
        loss = None
        if targets is not None:
            b, s, v = logits.shape
            loss = self.loss_fn(
                logits.reshape(b * s, v),
                targets.reshape(b * s)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # Block size'ı aşarsa kırp
            current = input_ids[:, -self.block_size:]

            logits, _ = self.forward(current)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids



if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TelemetryGPT(
        vocab_size=2000,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        block_size=128
    ).to(device)

    # Dummy input
    x = torch.randint(0, 2000, (4, 127)).to(device)
    y = torch.randint(0, 2000, (4, 127)).to(device)

    logits, loss = model(x, y)
    print(f"Input  : {x.shape}")  # [4, 127]
    print(f"Logits : {logits.shape}")  # [4, 127, 2000]
    print(f"Loss   : {loss.item():.4f}")