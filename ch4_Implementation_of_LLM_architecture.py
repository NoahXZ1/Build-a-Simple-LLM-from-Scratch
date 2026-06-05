"""This is for chapter 4, Implementation of LLM architecture."""
# This is the configuration for a small GPT-2 model with 124 mill parameters. 
GPT_CONFIG_124M = {
    "vocab_size": 50257,  #vocabulary size
    "context_length": 1024,  #context_length
    "emb_dim": 768, #Embedding dimension
    "n_heads": 12,  #number of attention heads
    "n_layers": 12,  #number of transformer blocks
    "drop_rate": 0.1,  #Dropout rate
    "qkv_bias": False,  #Query-Key-Value bias
}

#implement a placeholder GPT model architecture class
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #use a placeholder for transformer blocks
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )
        #use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

#a simple placeholder that will be replaced by a real TransformerBlock later
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

#a placeholder for LayerNorm
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
#tokenizer a batch consisting of two text inputs for the GPT model using the tiktoken tokenizer from chp2
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
#then we initialize a new 124M parameter DummyGPTModel instance and feed it the tokenized batch:
torch.manual_seed(123)
model=DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
#the output shape is (2,6,50257) which is (patch_size, seq_len, vocab_size)
print(logits) 